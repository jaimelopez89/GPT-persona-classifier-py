"""Hubspot API client for downloading reports and importing persona classifications.

This module provides functions to:
- Pull contact data from Hubspot reports or lists via API
- Import persona classifications back into Hubspot as contact properties

Author: Jaime López, 2025
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Path to persona mapping file
PERSONA_MAPPING_FILE = Path(__file__).parent / "hubspot_persona_mapping.json"


def get_hubspot_api_key() -> Optional[str]:
    """Get Hubspot API key from environment variables.

    Returns:
        Hubspot API key if set, None otherwise.
    """
    return os.getenv("HUBSPOT_API_KEY")


def _load_persona_mapping() -> dict[str, str]:
    """Load persona name to Hubspot enum mapping from external JSON file.

    Returns:
        Dictionary mapping persona names to Hubspot enum values.

    Raises:
        RuntimeError: If mapping file cannot be loaded or is invalid.
    """
    if not PERSONA_MAPPING_FILE.exists():
        raise RuntimeError(
            f"Persona mapping file not found: {PERSONA_MAPPING_FILE}. "
            "Please create this file with the correct persona mappings."
        )

    try:
        with open(PERSONA_MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            mapping = data.get("persona_mapping", {})
            if not mapping:
                raise ValueError("persona_mapping is empty or missing")
            return mapping
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON in persona mapping file {PERSONA_MAPPING_FILE}: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error loading persona mapping file {PERSONA_MAPPING_FILE}: {e}"
        ) from e


def map_persona_to_hubspot_enum(persona: str) -> str:
    """Map persona name to Hubspot enum value.

    Hubspot expects enum values like persona_1, persona_2, etc.
    This function maps the persona names to the corresponding enum values
    using the external mapping file.

    Args:
        persona: Persona name (e.g., "Application Developer").

    Returns:
        Hubspot enum value (e.g., "persona_1"). If no mapping is found,
        returns the original persona name (which will likely cause an error
        in Hubspot, but allows the user to see what went wrong).

    Note:
        This mapping is only used when uploading to Hubspot via API.
        Human-readable output files still use the original persona names.
    """
    persona_clean = str(persona).strip()
    
    try:
        mapping = _load_persona_mapping()
        # Try exact match first
        if persona_clean in mapping:
            return mapping[persona_clean]
        
        # Try case-insensitive match
        for key, value in mapping.items():
            if key.lower() == persona_clean.lower():
                return value
        
        # No match found - return original (will cause error but shows what's wrong)
        return persona_clean
    except RuntimeError as e:
        # If mapping file can't be loaded, raise the error
        raise RuntimeError(
            f"Cannot map persona '{persona_clean}' to Hubspot enum: {e}"
        ) from e


def pull_list_contacts(
    list_id: str, limit: int = 10000
) -> pd.DataFrame:
    """Pull contacts from a Hubspot list (segment).

    Fetches contacts from the specified Hubspot list/segment using the Lists API.
    Handles pagination automatically.

    Args:
        list_id: Hubspot list/segment ID (required). Must be a valid list ID
            that the API key has access to.
        limit: Maximum number of contacts to fetch (default: 10000).

    Returns:
        DataFrame with columns: Prospect Id, Email, Job Title, First Name,
        Last Name, Company.

    Raises:
        ValueError: If list_id is None or empty.
        RuntimeError: If HUBSPOT_API_KEY is not set, or if list cannot be
            accessed or processed.
        requests.HTTPError: If API request fails with detailed error message.
    """
    # Validate list_id is provided
    if not list_id or not str(list_id).strip():
        raise ValueError(
            "list_id is required. Cannot fetch all contacts. "
            "Please provide a valid Hubspot list/segment ID."
        )

    list_id = str(list_id).strip()
    api_key = get_hubspot_api_key()
    if not api_key:
        raise RuntimeError(
            "HUBSPOT_API_KEY not set. Please set it in your .env file or "
            "environment variables."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Properties to fetch from Hubspot
    properties = [
        "email", "jobtitle", "firstname", "lastname", "hs_object_id", "company"
    ]

    # Use Lists API v1 to fetch contacts
    # Endpoint: GET /contacts/v1/lists/{list_id}/contacts/all
    url = f"https://api.hubapi.com/contacts/v1/lists/{list_id}/contacts/all"
    
    print(f"Fetching contacts from Hubspot list/segment ID: {list_id}...")

    all_contacts = []
    vid_offset = None
    fetched = 0

    while fetched < limit:
        # Lists API v1 uses property parameter as comma-separated string
        params = {
            "property": ",".join(properties),
            "count": min(100, limit - fetched)  # Hubspot default is 20, max is 100
        }
        
        if vid_offset:
            params["vidOffset"] = vid_offset

        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=120
            )
            response.raise_for_status()
            data = response.json()

            contacts = data.get("contacts", [])
            if not contacts:
                break

            all_contacts.extend(contacts)
            fetched += len(contacts)

            print(f"Fetched {fetched} contacts...")

            # Check if there are more contacts
            has_more = data.get("has-more", False)
            if not has_more:
                break

            vid_offset = data.get("vid-offset")
            if not vid_offset:
                break

        except requests.HTTPError as e:
            error_detail = ""
            if e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("message", "")
                    if not error_detail:
                        error_detail = str(error_json)
                except (ValueError, KeyError):
                    error_detail = e.response.text[:200]

            status_code = e.response.status_code if e.response else "unknown"

            if status_code == 404:
                raise RuntimeError(
                    f"Hubspot list/segment not found: List ID '{list_id}' does not exist "
                    f"or you do not have access to it. "
                    f"Please verify the list ID is correct and that your API key "
                    f"has permission to access this list. "
                    f"Error details: {error_detail}"
                ) from e
            elif status_code == 403:
                raise RuntimeError(
                    f"Access denied to Hubspot list/segment '{list_id}'. "
                    f"Your API key does not have permission to access this list. "
                    f"Please check your Hubspot private app permissions. "
                    f"Error details: {error_detail}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to fetch contacts from Hubspot list '{list_id}': "
                    f"HTTP {status_code}. {error_detail}"
                ) from e

    # Convert to DataFrame
    # Hubspot Lists API v1 returns properties in format: {"property": {"value": "..."}}
    rows = []
    for contact in all_contacts:
        props = contact.get("properties", {})
        vid = contact.get("vid") or contact.get("canonical-vid", "")
        
        # Helper to extract property value (handles both v1 and v3 formats)
        def get_prop_value(prop_name: str) -> str:
            prop = props.get(prop_name, {})
            if isinstance(prop, dict) and "value" in prop:
                return str(prop.get("value", ""))
            elif isinstance(prop, str):
                return prop
            else:
                return ""

        email = get_prop_value("email")
        jobtitle = get_prop_value("jobtitle")
        firstname = get_prop_value("firstname")
        lastname = get_prop_value("lastname")
        company = get_prop_value("company")

        rows.append({
            "Prospect Id": str(vid),
            "Email": email,
            "Job Title": jobtitle,
            "First Name": firstname,
            "Last Name": lastname,
            "Company": company
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(
            f"Warning: No contacts found for list/segment '{list_id}'. "
            "The list may be empty or the filters may not match any contacts."
        )
    else:
        print(f"Successfully fetched {len(df)} contacts from Hubspot list/segment")
    return df


def pull_report_contacts(
    report_id: str, limit: int = 10000
) -> pd.DataFrame:
    """Pull contacts from a Hubspot report.

    Fetches contacts from the specified Hubspot report by first retrieving
    the report definition, then using the Contacts Search API with appropriate
    filters. Requires a valid report ID.

    Args:
        report_id: Hubspot report ID (required). Must be a valid report ID
            that the API key has access to.
        limit: Maximum number of contacts to fetch (default: 10000).

    Returns:
        DataFrame with columns: Prospect Id, Email, Job Title, First Name,
        Last Name, Company.

    Raises:
        ValueError: If report_id is None or empty.
        RuntimeError: If HUBSPOT_API_KEY is not set, or if report cannot be
            accessed or processed.
        requests.HTTPError: If API request fails with detailed error message.
    """
    # Validate report_id is provided
    if not report_id or not str(report_id).strip():
        raise ValueError(
            "report_id is required. Cannot fetch all contacts. "
            "Please provide a valid Hubspot report ID."
        )

    report_id = str(report_id).strip()
    api_key = get_hubspot_api_key()
    if not api_key:
        raise RuntimeError(
            "HUBSPOT_API_KEY not set. Please set it in your .env file or "
            "environment variables."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Properties to fetch from Hubspot
    properties = [
        "email", "jobtitle", "firstname", "lastname", "hs_object_id", "company"
    ]

    # First, try to fetch the report definition to validate it exists
    # and get any associated list IDs or filters
    # Try multiple possible endpoints in case the API structure has changed
    report_urls = [
        f"https://api.hubapi.com/reports/v3/reports/{report_id}",
        f"https://api.hubapi.com/reports/v2/reports/{report_id}",
        f"https://api.hubapi.com/analytics/v3/reports/{report_id}",
    ]
    
    report_data = None
    
    for report_url in report_urls:
        print(f"Trying to fetch report from: {report_url}...")
        try:
            report_response = requests.get(
                report_url, headers=headers, timeout=120
            )
            report_response.raise_for_status()
            report_data = report_response.json()
            print(f"✓ Report found via {report_url}")
            print(f"  Report name: {report_data.get('name', 'Unknown')}")
            break
        except requests.HTTPError as e:
            if e.response and e.response.status_code == 404:
                # Try next URL
                print(f"  Not found at {report_url} (404), trying next endpoint...")
                continue
            else:
                # Other errors - show details but continue trying other endpoints
                status_code = e.response.status_code if e.response else "unknown"
                error_detail = ""
                if e.response:
                    try:
                        error_json = e.response.json()
                        error_detail = error_json.get("message", str(error_json)[:100])
                    except (ValueError, KeyError):
                        error_detail = e.response.text[:100]
                
                print(
                    f"  Error at {report_url}: HTTP {status_code} - {error_detail}"
                )
                # If it's a 403 (forbidden), might be permission issue, try next
                if e.response and e.response.status_code == 403:
                    print("  Access denied, trying next endpoint...")
                    continue
                # For other errors, still try next endpoint but note it
                if report_url == report_urls[-1]:
                    # Last endpoint failed, raise the error
                    raise
                continue
    
    if report_data is None:
        # None of the endpoints worked, raise error with details
        raise RuntimeError(
            f"Report '{report_id}' not found at any API endpoint.\n\n"
            f"Tried endpoints:\n"
            f"  - /reports/v3/reports/{report_id}\n"
            f"  - /reports/v2/reports/{report_id}\n"
            f"  - /analytics/v3/reports/{report_id}\n\n"
            f"Possible issues:\n"
            f"  1. Report ID format is incorrect\n"
            f"  2. Report ID is from URL but needs different format\n"
            f"  3. API endpoint has changed\n\n"
            f"To find the correct report ID:\n"
            f"  - In Hubspot, open the report\n"
            f"  - Check the URL: it should contain a numeric ID\n"
            f"  - The ID might be in format like: /reports/{report_id}/...\n"
            f"  - Or check the report's 'Settings' or 'Properties' for the ID\n"
        )

    # Try to extract list IDs from the report if available
    # Hubspot reports can be associated with lists
    list_ids = []
    if "objectId" in report_data:
        # Some reports have direct object IDs
        pass
    if "listIds" in report_data:
        list_ids = report_data.get("listIds", [])
    
    # Debug: print report structure to help diagnose issues
    print(f"Report structure keys: {list(report_data.keys())[:10]}...")

    # Now fetch contacts using the Contacts Search API
    # If the report has associated lists, we could filter by list membership
    # For now, we'll fetch contacts and note that report-based filtering
    # may need to be done via lists
    all_contacts = []
    after = None
    url = "https://api.hubapi.com/crm/v3/objects/contacts/search"

    # Build filter groups - if we have list IDs, we can filter by them
    filter_groups = []
    if list_ids:
        # Filter by list membership
        for list_id in list_ids:
            filter_groups.append({
                "filters": [{
                    "propertyName": "hs_list_membership",
                    "operator": "IN",
                    "value": list_id
                }]
            })
        print(f"Filtering contacts by list IDs: {list_ids}")
    else:
        # Note: Hubspot Reports API doesn't directly expose contact filters
        # in a way we can easily translate to the Search API.
        # The report may use complex criteria that aren't directly mappable.
        print(
            f"Note: Report '{report_id}' does not have associated list IDs. "
            "Fetching all contacts. "
            "For report-specific filtering, consider exporting the report "
            "as a CSV or associating the report with a contact list."
        )

    payload = {
        "filterGroups": filter_groups,
        "properties": properties,
        "limit": min(100, limit)  # Hubspot max per request is 100
    }

    print("Fetching contacts from Hubspot...")
    fetched = 0

    while fetched < limit:
        if after:
            payload["after"] = after

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            all_contacts.extend(results)
            fetched += len(results)

            print(f"Fetched {fetched} contacts...")

            paging = data.get("paging")
            if not paging or not paging.get("next"):
                break
            after = paging["next"].get("after")
            if not after:
                break

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(e.response.headers.get("Retry-After", 10))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            # Provide detailed error for contact search failures
            error_detail = ""
            if e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("message", "")
                    if not error_detail:
                        error_detail = str(error_json)
                except (ValueError, KeyError):
                    error_detail = e.response.text[:200]

            status_code = e.response.status_code if e.response else "unknown"
            raise RuntimeError(
                f"Failed to fetch contacts for report '{report_id}': "
                f"HTTP {status_code}. {error_detail}"
            ) from e

    # Convert to DataFrame
    rows = []
    for contact in all_contacts:
        props = contact.get("properties", {})
        rows.append({
            "Prospect Id": str(props.get("hs_object_id", "")),
            "Email": props.get("email", ""),
            "Job Title": props.get("jobtitle", ""),
            "First Name": props.get("firstname", ""),
            "Last Name": props.get("lastname", ""),
            "Company": props.get("company", "")
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(
            f"Warning: No contacts found for report '{report_id}'. "
            "The report may be empty or the filters may not match any contacts."
        )
    else:
        print(f"Successfully fetched {len(df)} contacts from Hubspot report")
    return df


def _lookup_contact_ids_by_emails(
    emails: list[str], headers: dict
) -> dict[str, str]:
    """Look up Hubspot contact IDs by email addresses.

    Uses the Contacts Search API to find contacts by email.

    Args:
        emails: List of email addresses to look up.
        headers: HTTP headers for API requests.

    Returns:
        Dictionary mapping email (lowercase) -> contact_id.
    """
    email_to_id = {}
    url = "https://api.hubapi.com/crm/v3/objects/contacts/search"
    
    # Process emails in batches (Hubspot allows up to 100 OR filters per request)
    # We'll use IN operator to search for multiple emails at once
    batch_size = 100
    
    for i in range(0, len(emails), batch_size):
        email_batch = emails[i:i + batch_size]
        
        # Build filter with IN operator for multiple emails
        search_payload = {
            "filterGroups": [{
                "filters": [{
                    "propertyName": "email",
                    "operator": "IN",
                    "values": email_batch
                }]
            }],
            "properties": ["hs_object_id", "email"],
            "limit": len(email_batch)
        }

        try:
            response = requests.post(
                url, headers=headers, json=search_payload, timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            for result in data.get("results", []):
                props = result.get("properties", {})
                email_addr = props.get("email", "").strip().lower()
                contact_id = props.get("hs_object_id", "")
                if email_addr and contact_id:
                    email_to_id[email_addr] = str(contact_id)
                    
        except requests.HTTPError as e:
            # If batch search fails, try individual lookups
            if e.response and e.response.status_code == 400:
                # IN operator might not be supported, fall back to individual
                print("Batch lookup not supported, using individual lookups...")
                for email in email_batch:
                    try:
                        individual_payload = {
                            "filterGroups": [{
                                "filters": [{
                                    "propertyName": "email",
                                    "operator": "EQ",
                                    "value": email
                                }]
                            }],
                            "properties": ["hs_object_id", "email"],
                            "limit": 1
                        }
                        individual_response = requests.post(
                            url, headers=headers, json=individual_payload, timeout=120
                        )
                        individual_response.raise_for_status()
                        individual_data = individual_response.json()
                        results = individual_data.get("results", [])
                        if results:
                            props = results[0].get("properties", {})
                            contact_id = props.get("hs_object_id", "")
                            if contact_id:
                                email_to_id[email] = str(contact_id)
                    except (requests.HTTPError, requests.RequestException, KeyError, ValueError):
                        continue
            else:
                # Other errors, log and continue
                print(f"Warning: Error looking up emails: {e}")

    return email_to_id


def import_classified_contacts(
    df: pd.DataFrame,
    update_existing: bool = True,
    batch_size: int = 100
) -> dict:
    """Import persona classifications back into Hubspot contacts.

    Updates contact properties with Persona and Persona Certainty values.
    Matches contacts by email address (not Prospect ID).
    Uses Hubspot's batch update API for efficiency.

    Args:
        df: DataFrame with columns: Email, Persona, Persona Certainty.
            Prospect Id and Skip Reason columns are ignored.
        update_existing: Reserved for future use. Currently always updates
            existing contacts (default: True).
        batch_size: Number of contacts to update per batch (max 100,
            default: 100).

    Returns:
        Dictionary with import statistics:
        - successful: Number of successfully updated contacts
        - failed: Number of failed updates
        - total: Total number of contacts attempted
        - errors: List of error messages (if any)

    Raises:
        RuntimeError: If HUBSPOT_API_KEY is not set.
        ValueError: If required columns are missing from df.
    """
    # update_existing is reserved for future use (e.g., create vs update endpoints)
    _ = update_existing  # Suppress unused argument warning

    api_key = get_hubspot_api_key()
    if not api_key:
        raise RuntimeError(
            "HUBSPOT_API_KEY not set. Please set it in your .env file or "
            "environment variables."
        )

    # Validate required columns - now requires Email instead of Prospect Id
    required_cols = ["Email", "Persona"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Filter out rows with empty emails
    df = df[df["Email"].notna() & (df["Email"].astype(str).str.strip() != "")]
    if df.empty:
        print("Warning: No contacts with valid email addresses to import.")
        return {"successful": 0, "failed": 0, "total": 0, "errors": []}

    # Default property names (can be customized in Hubspot)
    persona_property = "hs_persona"
    certainty_property = "persona_certainty"

    url = "https://api.hubapi.com/crm/v3/objects/contacts/batch/update"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # First, look up contact IDs by email
    print(f"\nLooking up {len(df)} contacts by email address...")
    emails = df["Email"].astype(str).str.strip().str.lower().unique().tolist()
    email_to_id = _lookup_contact_ids_by_emails(emails, headers)
    
    if not email_to_id:
        raise RuntimeError(
            "No contacts found in Hubspot matching the provided email addresses. "
            "Please verify the emails are correct and the contacts exist in Hubspot."
        )
    
    print(f"Found {len(email_to_id)} matching contacts in Hubspot.")

    # Filter df to only include contacts we found
    df["Email_lower"] = df["Email"].astype(str).str.strip().str.lower()
    df = df[df["Email_lower"].isin(email_to_id.keys())]
    
    if df.empty:
        raise RuntimeError(
            "No matching contacts found after email lookup. "
            "Please verify the email addresses are correct."
        )

    successful = 0
    failed = 0
    errors = []

    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"\nImporting {len(df)} contacts to Hubspot in {total_batches} batch(es)...")

    for batch_num in range(0, len(df), batch_size):
        batch = df.iloc[batch_num:batch_num + batch_size]
        batch_idx = (batch_num // batch_size) + 1

        inputs = []
        for _, row in batch.iterrows():
            email = str(row["Email"]).strip().lower()
            contact_id = email_to_id.get(email)
            
            if not contact_id:
                failed += 1
                errors.append(f"Contact not found for email: {email}")
                continue

            # Map persona name to Hubspot enum value
            persona_name = str(row.get("Persona", "")).strip()
            persona_enum = map_persona_to_hubspot_enum(persona_name)

            properties = {
                persona_property: persona_enum
            }

            # Add certainty if present
            if "Persona Certainty" in row:
                certainty_value = str(row.get("Persona Certainty", "")).strip()
                if certainty_value:
                    properties[certainty_property] = certainty_value

            inputs.append({
                "id": contact_id,
                "properties": properties
            })

        if not inputs:
            continue

        payload = {"inputs": inputs}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            successful += len(inputs)
            print(
                f"Batch {batch_idx}/{total_batches}: "
                f"Successfully updated {len(inputs)} contacts"
            )

        except requests.HTTPError as e:
            failed += len(inputs)
            error_msg = f"Batch {batch_idx} failed: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {error_detail}"
                except ValueError:
                    error_msg += f" - {e.response.text[:200]}"
            errors.append(error_msg)
            print(f"ERROR: {error_msg}")

            # Handle rate limiting
            if e.response and e.response.status_code == 429:
                retry_after = int(
                    e.response.headers.get("Retry-After", 10)
                )
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)

        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            failed += len(inputs)
            error_msg = f"Batch {batch_idx} failed with unexpected error: {e}"
            errors.append(error_msg)
            print(f"ERROR: {error_msg}")

    result = {
        "successful": successful,
        "failed": failed,
        "total": len(df),
        "errors": errors
    }

    print(
        f"\nHubspot import complete: {successful} successful, "
        f"{failed} failed out of {len(df)} total"
    )

    return result

