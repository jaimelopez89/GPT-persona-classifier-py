"""Hubspot API client for downloading reports and importing persona classifications.

This module provides functions to:
- Pull contact data from Hubspot reports or lists via API
- Import persona classifications back into Hubspot as contact properties

Author: Jaime López, 2025
"""

import os
import time
from typing import Optional
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def get_hubspot_api_key() -> Optional[str]:
    """Get Hubspot API key from environment variables.

    Returns:
        Hubspot API key if set, None otherwise.
    """
    return os.getenv("HUBSPOT_API_KEY")


def pull_report_contacts(
    report_id: Optional[str] = None, limit: int = 10000
) -> pd.DataFrame:
    """Pull contacts from a Hubspot report or directly from contacts.

    If report_id is provided, attempts to fetch contacts from that report.
    Otherwise, fetches contacts directly using the Contacts Search API.

    Args:
        report_id: Optional Hubspot report ID. If None, pulls all contacts.
        limit: Maximum number of contacts to fetch (default: 10000).

    Returns:
        DataFrame with columns: Prospect Id, Email, Job Title, First Name,
        Last Name, Company.

    Raises:
        RuntimeError: If HUBSPOT_API_KEY is not set.
        requests.HTTPError: If API request fails.
    """
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

    all_contacts = []
    after = None
    url = "https://api.hubapi.com/crm/v3/objects/contacts/search"

    # Build filter if report_id is provided
    # Note: Hubspot Reports API doesn't directly return contacts, so we
    # use the Contacts Search API. If you have a specific list or filter,
    # you can modify the filterGroups below.
    filter_groups = []
    if report_id:
        # If report_id is provided, you may need to map it to a list ID
        # or use a different approach. For now, we'll search all contacts
        # and let the user filter by list if needed.
        print(
            f"Note: Report ID {report_id} provided. "
            "Fetching contacts directly (report-to-list mapping not implemented)."
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
            raise

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
    print(f"Successfully fetched {len(df)} contacts from Hubspot")
    return df


def import_classified_contacts(
    df: pd.DataFrame,
    update_existing: bool = True,
    batch_size: int = 100
) -> dict:
    """Import persona classifications back into Hubspot contacts.

    Updates contact properties with Persona and Persona Certainty values.
    Uses Hubspot's batch update API for efficiency.

    Args:
        df: DataFrame with columns: Prospect Id (or hs_object_id), Persona,
            Persona Certainty. Other columns are ignored.
        update_existing: If True, updates existing contacts. If False, only
            creates new contacts (default: True).
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
    api_key = get_hubspot_api_key()
    if not api_key:
        raise RuntimeError(
            "HUBSPOT_API_KEY not set. Please set it in your .env file or "
            "environment variables."
        )

    # Validate required columns
    required_cols = ["Prospect Id", "Persona"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Default property names (can be customized in Hubspot)
    persona_property = "persona"
    certainty_property = "persona_certainty"

    url = "https://api.hubapi.com/crm/v3/objects/contacts/batch/update"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

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
            prospect_id = str(row["Prospect Id"]).strip()
            if not prospect_id:
                continue

            properties = {
                persona_property: str(row.get("Persona", "")).strip()
            }

            # Add certainty if present
            if "Persona Certainty" in row:
                properties[certainty_property] = str(
                    row.get("Persona Certainty", "")
                ).strip()

            inputs.append({
                "id": prospect_id,
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

        except Exception as e:
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

