"""Hubspot API client for downloading reports and importing persona classifications.

This module provides functions to:
- Pull contact data from Hubspot reports or lists via API
- Import persona classifications back into Hubspot as contact properties

Author: Jaime López, 2025
"""

import os
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Path to persona mapping file
PERSONA_MAPPING_FILE = Path(__file__).parent / "hubspot_persona_mapping.json"


def get_hubspot_api_key() -> Optional[str]:
    """Get Hubspot API key from environment variables (for read operations).

    Returns:
        API key string if found, None otherwise.
    """
    return os.getenv("HUBSPOT_API_KEY")


def get_hubspot_write_token() -> Optional[str]:
    """Get Hubspot write API key from environment variables (for write operations).

    This key is used for importing/updating contacts in Hubspot.
    Uses HUBSPOT_WRITE_API_KEY from environment variables.

    Returns:
        Write API key string if found, None otherwise.
    """
    return os.getenv("HUBSPOT_WRITE_API_KEY")


def _load_persona_mapping() -> dict[str, str]:
    """Load persona name to Hubspot enum mapping from JSON file.

    Returns:
        Dictionary mapping persona names to Hubspot enum values.

    Raises:
        RuntimeError: If the mapping file cannot be loaded or parsed.
    """
    if not PERSONA_MAPPING_FILE.exists():
        raise RuntimeError(
            f"Persona mapping file not found: {PERSONA_MAPPING_FILE}. "
            "Please create this file with the persona name to enum mapping."
        )

    try:
        with open(PERSONA_MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            mapping = data.get("persona_mapping", {})
            if not mapping:
                raise RuntimeError(
                    f"No 'persona_mapping' key found in {PERSONA_MAPPING_FILE}"
                )
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


def _verify_list_exists(list_id: str, headers: dict) -> dict:
    """Verify that a list exists and return its metadata.
    
    Args:
        list_id: Hubspot list/segment ID.
        headers: Request headers including Authorization.
        
    Returns:
        Dictionary with list metadata if found.
        
    Raises:
        RuntimeError: If list is not found or not accessible.
    """
    # Try to get list metadata
    list_url = f"https://api.hubapi.com/contacts/v1/lists/{list_id}"
    
    try:
        response = requests.get(list_url, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response else "unknown"
        if status_code == 404:
            raise RuntimeError(
                f"List/segment '{list_id}' not found. "
                f"Please verify the list ID is correct."
            ) from e
        elif status_code == 403:
            raise RuntimeError(
                f"Access denied to list/segment '{list_id}'. "
                f"Your API key may not have permission to access this list."
            ) from e
        else:
            # For other errors, just log and continue
            print(f"Warning: Could not verify list metadata: HTTP {status_code}")
            return {}


def pull_list_contacts(
    list_id: str, limit: int = 10000
) -> pd.DataFrame:
    """Pull contacts from a Hubspot list (segment).

    First gets contact IDs from the list, then fetches contact details.

    Args:
        list_id: Hubspot list/segment ID (required). Must be a valid list ID
            that the API key has access to.
        limit: Maximum number of contacts to fetch (default: 10000).

    Returns:
        DataFrame with columns: Prospect Id, Email, Job Title, Company.

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

    # Properties to fetch from Hubspot (only what we need)
    properties = ["email", "jobtitle", "company", "hs_object_id"]
    
    print(f"Fetching contacts from Hubspot list/segment ID: {list_id}...")
    
    # First, verify the list exists and get metadata
    try:
        list_metadata = _verify_list_exists(list_id, headers)
        if list_metadata:
            list_name = list_metadata.get("name", "Unknown")
            list_size = list_metadata.get("metaData", {}).get("size", "Unknown")
            print(f"List found: '{list_name}' (size: {list_size})")
    except RuntimeError as e:
        print(f"Warning: {e}")
        # Continue anyway - the list might exist but metadata endpoint might not work
    
    # Step 1: Get contact IDs from list memberships using v3 API
    print("Fetching contact IDs from list memberships...")
    memberships_url = f"https://api.hubapi.com/crm/v3/lists/{list_id}/memberships"
    
    contact_ids = []
    after = None
    
    while len(contact_ids) < limit:
        params = {"limit": min(100, limit - len(contact_ids))}
        if after:
            params["after"] = after
        
        try:
            response = requests.get(
                memberships_url, headers=headers, params=params, timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                break
            
            # Extract contact IDs from memberships
            for result in results:
                contact_id = result.get("contactId") or result.get("recordId")
                if contact_id:
                    contact_ids.append(str(contact_id))
            
            print(f"Found {len(contact_ids)} contact IDs in list...")
            
            # Check pagination
            paging = data.get("paging", {})
            after = paging.get("next", {}).get("after")
            if not after:
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
                    f"Failed to fetch list memberships from Hubspot list '{list_id}': "
                    f"HTTP {status_code}. {error_detail}"
                ) from e
    
    if not contact_ids:
        print(f"Warning: No contacts found in list/segment '{list_id}'.")
        expected_columns = ["Prospect Id", "Email", "Job Title", "Company"]
        return pd.DataFrame(columns=expected_columns)
    
    # Step 2: Batch fetch contact details using v3 batch read API
    print(f"Fetching details for {len(contact_ids)} contacts...")
    batch_read_url = "https://api.hubapi.com/crm/v3/objects/contacts/batch/read"
    
    all_contacts = []
    batch_size = 100  # Hubspot allows up to 100 IDs per batch
    
    for i in range(0, len(contact_ids), batch_size):
        batch_ids = contact_ids[i:i + batch_size]
        batch_payload = {
            "properties": properties,
            "propertiesWithHistory": [],
            "idProperty": "hs_object_id",
            "inputs": [{"id": cid} for cid in batch_ids]
        }
        
        try:
            response = requests.post(
                batch_read_url, headers=headers, json=batch_payload, timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            
            # Debug: Show first contact structure
            # if i == 0 and results:
            #     print(f"DEBUG: First contact structure: {list(results[0].keys())}")
            #     print(f"DEBUG: First contact properties: {list(results[0].get('properties', {}).keys())}")
            
            for result in results:
                props = result.get("properties", {})
                # v3 batch read returns id in the result, and hs_object_id in properties
                contact_id = result.get("id", "") or props.get("hs_object_id", "")
                contact = {
                    "vid": contact_id,
                    "canonical-vid": contact_id,
                    "properties": props
                }
                all_contacts.append(contact)
            
            print(f"Fetched details for {len(all_contacts)} contacts...")
            
            # Debug: Save first 5 contacts immediately after first batch
            # if i == 0 and all_contacts:
            #     debug_contacts = []
            #     for j, contact in enumerate(all_contacts[:5], 1):
            #         props = contact.get("properties", {})
            #         # Prioritize hs_object_id from properties, then vid
            #         contact_id = props.get("hs_object_id", "") or contact.get("vid", "") or contact.get("canonical-vid", "N/A")
            #         
            #         # v3 API returns properties as simple key-value pairs
            #         email = props.get("email", "")
            #         jobtitle = props.get("jobtitle", "")
            #         company = props.get("company", "")
            #         
            #         debug_contact = {
            #             "Contact Number": j,
            #             "Contact ID": str(contact_id),
            #             "Email": email,
            #             "Job Title": jobtitle,
            #             "Company": company
            #         }
            #         debug_contacts.append(debug_contact)
            #     
            #     debug_file = os.path.join(
            #         os.getcwd(), f"hubspot_debug_contacts_{list_id}.csv"
            #     )
            #     debug_df = pd.DataFrame(debug_contacts)
            #     debug_df.to_csv(debug_file, index=False)
            #     print(f"\n=== DEBUG: First 5 contacts saved to: {debug_file} ===")
                
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
            print(
                f"Warning: Failed to fetch batch {i//batch_size + 1}: "
                f"HTTP {status_code}. {error_detail}. Continuing..."
            )
            # Continue with next batch even if one fails

    # Convert to DataFrame - v3 API returns properties as simple key-value pairs
    rows = []
    for contact in all_contacts:
        props = contact.get("properties", {})
        # Use hs_object_id if available, otherwise fall back to vid
        contact_id = props.get("hs_object_id", "") or contact.get("vid", "") or contact.get("canonical-vid", "")

        email = props.get("email", "")
        jobtitle = props.get("jobtitle", "")
        company = props.get("company", "")

        rows.append({
            "Prospect Id": str(contact_id),
            "Email": email if email else "",
            "Job Title": jobtitle if jobtitle else "",
            "Company": company if company else ""
        })

    # Always create DataFrame with expected columns, even if empty
    expected_columns = ["Prospect Id", "Email", "Job Title", "Company"]
    if not rows:
        df = pd.DataFrame(columns=expected_columns)
        print(
            f"Warning: No contacts found for list/segment '{list_id}'. "
            "The list may be empty or the filters may not match any contacts."
        )
    else:
        df = pd.DataFrame(rows)
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
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

            for result in results:
                contact = {
                    "vid": result.get("id", ""),
                    "canonical-vid": result.get("id", ""),
                    "properties": result.get("properties", {})
                }
                all_contacts.append(contact)

            fetched += len(results)
            print(f"Fetched {fetched} contacts...")

            paging = data.get("paging", {})
            after = paging.get("next", {}).get("after")
            if not after:
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
            raise RuntimeError(
                f"Failed to fetch contacts from Hubspot report '{report_id}': "
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
    api_key: str, emails: list[str], headers: dict
) -> dict[str, str]:
    """Look up Hubspot contact IDs by email addresses.

    Uses the Contacts Search API to find contacts by email.

    Args:
        api_key: Hubspot API key.
        emails: List of email addresses to look up.
        headers: Request headers including Authorization.

    Returns:
        Dictionary mapping email addresses to Hubspot contact IDs.
    """
    email_to_id = {}
    url = "https://api.hubapi.com/crm/v3/objects/contacts/search"
    
    # Hubspot search API: multiple filters in one filterGroup = AND logic
    # To search for multiple emails (OR logic), we need separate filterGroups
    # However, Hubspot limits filterGroups, so we'll do individual searches
    # which is more reliable and simpler
    
    print(f"Looking up {len(emails)} contacts by email (this may take a moment)...")
    for idx, email in enumerate(emails, 1):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(emails)} emails...")
        
        try:
            search_payload = {
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
            
            search_response = requests.post(
                url, headers=headers, json=search_payload, timeout=120
            )
            search_response.raise_for_status()
            search_data = search_response.json()
            results = search_data.get("results", [])
            
            if results:
                props = results[0].get("properties", {})
                contact_id = props.get("hs_object_id", "")
                found_email = props.get("email", "")
                if contact_id:
                    # Use the email from the response to handle case sensitivity
                    email_to_id[found_email.lower()] = str(contact_id)
                    # Also map the original email (case-insensitive lookup)
                    if found_email.lower() != email.lower():
                        email_to_id[email.lower()] = str(contact_id)
        except requests.HTTPError as e:
            # Log but continue - some emails might not exist
            if e.response and e.response.status_code != 404:
                # Only log non-404 errors (404 means contact not found, which is OK)
                error_detail = ""
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("message", str(error_json)[:100])
                except (ValueError, KeyError):
                    error_detail = e.response.text[:100] if e.response else ""
                print(f"  Warning: Failed to lookup email {email}: {error_detail}")
            continue
        except Exception as e:
            # Skip any other errors and continue
            continue

    return email_to_id


def import_classified_contacts(
    df: pd.DataFrame,
    persona_property: str = "hs_persona",
    certainty_property: str = "persona_certainty",
) -> dict[str, int]:
    """Import classified personas back into Hubspot.

    Takes a DataFrame with classified personas and updates the corresponding
    Hubspot contacts. Matches contacts by email address.

    Args:
        df: DataFrame with columns: Email, Persona, Persona Certainty.
            May also contain Prospect Id and Skip Reason, which will be ignored.
        persona_property: Hubspot property name for persona (default: "hs_persona").
        certainty_property: Hubspot property name for certainty (default: "persona_certainty").

    Returns:
        Dictionary with keys: "success", "failed", "not_found" indicating
        counts of contacts updated, failed, and not found respectively.

    Raises:
        RuntimeError: If HUBSPOT_API_KEY is not set, or if required columns
            are missing from the DataFrame.
        ValueError: If DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Nothing to import.")

    required_columns = ["Email", "Persona"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"DataFrame missing required columns: {missing_columns}. "
            f"Required: {required_columns}"
        )

    # Use write API key for import/update operations
    write_token = get_hubspot_write_token()
    if not write_token:
        raise RuntimeError(
            "HUBSPOT_WRITE_API_KEY not set. Please set it in your .env file or "
            "environment variables. This key is required for importing/updating contacts."
        )

    headers = {
        "Authorization": f"Bearer {write_token}",
        "Content-Type": "application/json"
    }

    # Filter out rows with missing emails or personas
    df_clean = df[df["Email"].notna() & df["Persona"].notna()].copy()
    if df_clean.empty:
        raise ValueError(
            "No valid contacts to import. All contacts are missing Email or Persona."
        )

    # Prepare batch update payloads
    # Hubspot allows up to 100 contacts per batch for bulk updates
    batch_size = 100
    all_inputs = []
    
    # Check if we have Prospect Id column - if so, use it directly
    has_prospect_id = "Prospect Id" in df_clean.columns
    
    # Separate contacts with IDs from those needing email lookup
    contacts_with_id = []
    contacts_needing_lookup = []
    
    for _, row in df_clean.iterrows():
        prospect_id = None
        if has_prospect_id:
            prospect_id_val = row.get("Prospect Id")
            if pd.notna(prospect_id_val) and str(prospect_id_val).strip():
                prospect_id = str(prospect_id_val).strip()
        
        if prospect_id:
            contacts_with_id.append((prospect_id, row))
        else:
            contacts_needing_lookup.append(row)
    
    print(f"Found {len(contacts_with_id)} contacts with existing IDs")
    print(f"Need to lookup {len(contacts_needing_lookup)} contacts by email")
    
    # Look up contact IDs by email for contacts that don't have IDs
    # Use read API key for lookups (read operation)
    read_api_key = get_hubspot_api_key()
    if not read_api_key:
        raise RuntimeError(
            "HUBSPOT_API_KEY not set. Please set it in your .env file or "
            "environment variables. This key is required for looking up contacts."
        )
    
    email_to_id = {}
    if contacts_needing_lookup:
        emails = [str(row["Email"]).strip() for row in contacts_needing_lookup]
        emails = list(set(emails))  # Get unique emails
        # Use read API key for lookup operations
        lookup_headers = {
            "Authorization": f"Bearer {read_api_key}",
            "Content-Type": "application/json"
        }
        email_to_id = _lookup_contact_ids_by_emails(read_api_key, emails, lookup_headers)
        print(f"Found {len(email_to_id)} contacts in Hubspot via email lookup")

    # Process all contacts
    for prospect_id, row in contacts_with_id:
        # Use the existing Prospect Id
        contact_id = prospect_id
        all_inputs.append({
            "contact_id": contact_id,
            "row": row
        })
    
    for row in contacts_needing_lookup:
        email = str(row["Email"]).strip()
        # Use case-insensitive lookup
        contact_id = email_to_id.get(email.lower())
        
        if not contact_id:
            continue  # Skip contacts not found in Hubspot
        
        all_inputs.append({
            "contact_id": contact_id,
            "row": row
        })

    if not all_inputs:
        print("No contacts to update (none found in Hubspot).")
        total_contacts = len(contacts_with_id) + len(contacts_needing_lookup)
        return {"success": 0, "failed": 0, "not_found": total_contacts}

    # Build the actual update payloads
    update_inputs = []
    for item in all_inputs:
        contact_id = item["contact_id"]
        row = item["row"]

        # Map persona to Hubspot enum value
        persona = str(row["Persona"]).strip()
        persona_enum = map_persona_to_hubspot_enum(persona)

        # Build properties object
        properties = {
            persona_property: persona_enum
        }

        # Add certainty if available
        if "Persona Certainty" in row and pd.notna(row["Persona Certainty"]):
            properties[certainty_property] = str(row["Persona Certainty"])

        update_inputs.append({
            "id": contact_id,
            "properties": properties
        })

    if not update_inputs:
        print("No contacts to update (none found in Hubspot).")
        total_contacts = len(contacts_with_id) + len(contacts_needing_lookup)
        return {"success": 0, "failed": 0, "not_found": total_contacts}

    # Bulk update contacts using batch API
    # This processes contacts in batches of 100 (HubSpot's maximum)
    url = "https://api.hubapi.com/crm/v3/objects/contacts/batch/update"
    success_count = 0
    failed_count = 0
    total_batches = (len(update_inputs) + batch_size - 1) // batch_size
    
    print(f"\nBulk importing {len(update_inputs)} contacts in {total_batches} batch(es)...")
    print(f"Batch size: {batch_size} contacts per batch\n")

    for i in range(0, len(update_inputs), batch_size):
        batch = update_inputs[i:i + batch_size]
        payload = {"inputs": batch}
        batch_num = i // batch_size + 1

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            success_count += len(batch)
            print(f"✓ Batch {batch_num}/{total_batches}: Successfully updated {len(batch)} contacts")
        except requests.HTTPError as e:
            failed_count += len(batch)
            error_msg = f"Batch {batch_num}/{total_batches} failed: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {error_detail}"
                    
                    # Check for missing scopes error and provide helpful message
                    if e.response.status_code == 403:
                        errors = error_detail.get("errors", [])
                        for error in errors:
                            if "requiredGranularScopes" in error.get("context", {}):
                                required_scopes = error["context"]["requiredGranularScopes"]
                                print(f"\n⚠️  PERMISSION ERROR: Missing required HubSpot API scopes")
                                print(f"   Your API key needs the following scopes to update contacts:")
                                for scope in required_scopes:
                                    print(f"     - {scope}")
                                print(f"\n   To fix this:")
                                print(f"   1. Go to your HubSpot account settings")
                                print(f"   2. Navigate to Integrations > Private Apps")
                                print(f"   3. Edit your private app")
                                print(f"   4. Add the required scopes listed above")
                                print(f"   5. Save and regenerate your API key if needed")
                                print(f"\n   More info: https://developers.hubspot.com/scopes\n")
                                break
                except (ValueError, KeyError):
                    error_msg += f" - {e.response.text[:200]}"
            print(f"Error: {error_msg}")
            # Continue with next batch even if one fails
        except Exception as e:
            failed_count += len(batch)
            print(f"✗ Batch {batch_num}/{total_batches} error: {e}")

    # Calculate not_found count
    not_found_count = len(contacts_needing_lookup) - len(email_to_id)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Bulk Import Summary:")
    print(f"  Successfully updated: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Total processed: {success_count + failed_count + not_found_count}")
    print(f"{'='*60}\n")

    return {
        "success": success_count,
        "failed": failed_count,
        "not_found": not_found_count
    }
