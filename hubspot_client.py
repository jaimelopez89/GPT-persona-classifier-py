"""Hubspot API client for pulling report data and importing classified contacts.

This module provides functions to:
- Pull contact data from Hubspot reports or directly from contacts
- Import classified persona results back into Hubspot as contact properties

Author: Jaime López, 2025
"""

import os
import time
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

try:
    from hubspot import HubSpot
    from hubspot.crm.contacts import ApiException as ContactsApiException
    HUBSPOT_AVAILABLE = True
except ImportError:
    HubSpot = None  # type: ignore[assignment, misc]
    ContactsApiException = None  # type: ignore[assignment, misc]
    HUBSPOT_AVAILABLE = False

load_dotenv()


def get_hubspot_client() -> Optional[HubSpot]:
    """Initialize and return Hubspot API client.

    Uses HUBSPOT_ACCESS_TOKEN from environment variables.

    Returns:
        HubSpot client instance, or None if token not found or library not available.

    Note:
        Requires hubspot-api-client package and HUBSPOT_ACCESS_TOKEN environment variable.
        Get your access token from: https://app.hubspot.com/private-apps
    """
    if not HUBSPOT_AVAILABLE:
        return None

    access_token = os.getenv("HUBSPOT_ACCESS_TOKEN")
    if not access_token:
        return None

    return HubSpot(access_token=access_token)


def pull_report_contacts(report_id: Optional[str] = None, limit: int = 10000) -> pd.DataFrame:
    """Pull contact data from Hubspot, either from a report or directly.

    If report_id is provided, attempts to fetch contacts from that report.
    Otherwise, fetches contacts directly (you can add filters as needed).

    Args:
        report_id: Optional Hubspot report ID. If None, fetches contacts directly.
        limit: Maximum number of contacts to retrieve (default: 10000).

    Returns:
        DataFrame with contact data, normalized to match expected input format.
        Columns: Prospect Id, Email, Job Title, First Name, Last Name, Company

    Raises:
        RuntimeError: If Hubspot client is not available or API call fails.
        ValueError: If no contacts found.
    """
    client = get_hubspot_client()
    if not client:
        raise RuntimeError(
            "Hubspot client not available. Install hubspot-api-client and set "
            "HUBSPOT_ACCESS_TOKEN in environment variables."
        )

    contacts_data = []
    properties = ['email', 'firstname', 'lastname', 'jobtitle', 'company', 'hs_object_id']

    try:
        if report_id:
            # Try to get contacts from report
            # Note: Hubspot Reports API may require different approach based on report type
            # This is a simplified implementation
            print(f"Fetching contacts from report {report_id}...")
            try:
                # Get report results
                results = client.crm.reports.reports_api.get_report_results(
                    report_id=report_id,
                    limit=limit
                )

                # Extract contact IDs
                contact_ids = []
                if hasattr(results, 'results') and results.results:
                    for result in results.results:
                        if hasattr(result, 'object_id'):
                            contact_ids.append(result.object_id)
                        elif hasattr(result, 'id'):
                            contact_ids.append(result.id)

                # Fetch contact details
                for contact_id in contact_ids[:limit]:
                    try:
                        contact = client.crm.contacts.basic_api.get_by_id(
                            contact_id=contact_id,
                            properties=properties
                        )
                        props = contact.properties
                        contacts_data.append({
                            'Prospect Id': props.get('hs_object_id', contact_id),
                            'Email': props.get('email', ''),
                            'Job Title': props.get('jobtitle', ''),
                            'First Name': props.get('firstname', ''),
                            'Last Name': props.get('lastname', ''),
                            'Company': props.get('company', ''),
                        })
                        time.sleep(0.1)  # Rate limiting
                    except ContactsApiException as e:
                        print(f"Warning: Could not fetch contact {contact_id}: {e}")
                        continue

            except (ValueError, AttributeError, KeyError) as e:
                print(f"Note: Could not access report directly ({e}), fetching all contacts...")
                report_id = None  # Fall back to direct fetch

        if not report_id or not contacts_data:
            # Fetch contacts directly
            print("Fetching contacts directly from Hubspot...")
            all_contacts = []
            after = None

            while len(all_contacts) < limit:
                response = client.crm.contacts.basic_api.get_page(
                    limit=min(100, limit - len(all_contacts)),
                    properties=properties,
                    after=after
                )
                all_contacts.extend(response.results)
                if not response.paging or not response.paging.next:
                    break
                after = response.paging.next.after
                time.sleep(0.1)  # Rate limiting

            for contact in all_contacts:
                props = contact.properties
                contacts_data.append({
                    'Prospect Id': props.get('hs_object_id', contact.id),
                    'Email': props.get('email', ''),
                    'Job Title': props.get('jobtitle', ''),
                    'First Name': props.get('firstname', ''),
                    'Last Name': props.get('lastname', ''),
                    'Company': props.get('company', ''),
                })

        if not contacts_data:
            raise ValueError("No contacts found")

        df = pd.DataFrame(contacts_data)

        # Normalize column names
        if "Record ID" in df.columns and "Prospect Id" not in df.columns:
            df = df.rename(columns={"Record ID": "Prospect Id"})

        print(f"Successfully fetched {len(df)} contacts from Hubspot")
        return df

    except ContactsApiException as e:
        raise RuntimeError(f"Hubspot Contacts API error: {e}") from e
    except (ValueError, AttributeError, KeyError) as e:
        raise RuntimeError(f"Error pulling Hubspot data: {e}") from e


def import_classified_contacts(df: pd.DataFrame, update_existing: bool = True, batch_size: int = 100) -> dict:
    """Import classified persona results into Hubspot as contact properties.

    Takes a DataFrame with classified personas and updates Hubspot contacts.
    Excludes "Prospect Id" and "Skip Reason" columns from import.
    Maps remaining columns to Hubspot contact properties.

    Args:
        df: DataFrame with classified results. Must contain "Prospect Id" or "Email"
            to identify contacts, plus persona classification columns.
        update_existing: If True, update existing contacts. If False, only create new ones.
        batch_size: Number of contacts to process in each batch (default: 100).

    Returns:
        Dictionary with import statistics:
        {
            'created': number of contacts created,
            'updated': number of contacts updated,
            'failed': number of failed operations,
            'errors': list of error messages (first 10)
        }

    Raises:
        RuntimeError: If Hubspot client is not available.
        ValueError: If DataFrame doesn't contain required identifier columns.
    """
    client = get_hubspot_client()
    if not client:
        raise RuntimeError(
            "Hubspot client not available. Install hubspot-api-client and set "
            "HUBSPOT_ACCESS_TOKEN in environment variables."
        )

    # Validate required columns
    if "Prospect Id" not in df.columns and "Email" not in df.columns:
        raise ValueError("DataFrame must contain either 'Prospect Id' or 'Email' column")

    # Columns to exclude from import
    exclude_cols = {"Prospect Id", "Skip Reason", "Record ID"}

    # Get columns to import (all except excluded ones)
    import_cols = [col for col in df.columns if col not in exclude_cols]

    if not import_cols:
        raise ValueError("No columns to import after excluding 'Prospect Id' and 'Skip Reason'")

    # Map common column names to Hubspot property names
    # Note: You may need to create custom properties "persona" and "persona_certainty" in Hubspot
    property_mapping = {
        "Persona": "persona",
        "Persona Certainty": "persona_certainty",
        "Email": "email",
        "First Name": "firstname",
        "Last Name": "lastname",
        "Job Title": "jobtitle",
        "Company": "company",
    }

    stats = {
        'created': 0,
        'updated': 0,
        'failed': 0,
        'errors': []
    }

    print(f"Importing {len(df)} classified contacts to Hubspot...")

    # Process in batches
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        for idx, row in batch_df.iterrows():
            try:
                # Get contact identifier
                contact_id = None
                email = None

                if "Prospect Id" in df.columns and pd.notna(row.get("Prospect Id")):
                    contact_id = str(row["Prospect Id"]).strip()
                elif "Email" in df.columns and pd.notna(row.get("Email")):
                    email = str(row["Email"]).strip()
                else:
                    stats['failed'] += 1
                    if len(stats['errors']) < 10:
                        stats['errors'].append(f"Row {idx}: No valid identifier")
                    continue

                # Build properties dictionary
                properties = {}
                for col in import_cols:
                    value = row.get(col)
                    if pd.notna(value) and str(value).strip():
                        # Map column name to Hubspot property name
                        prop_name = property_mapping.get(col, col.lower().replace(" ", "_"))
                        properties[prop_name] = str(value).strip()

                if not properties:
                    stats['failed'] += 1
                    if len(stats['errors']) < 10:
                        stats['errors'].append(f"Row {idx}: No properties to update")
                    continue

                # Create or update contact
                try:
                    if contact_id:
                        # Update by ID
                        client.crm.contacts.basic_api.update(
                            contact_id=contact_id,
                            simple_public_object_input={"properties": properties}
                        )
                        stats['updated'] += 1
                    elif email:
                        # Try to find existing contact by email
                        try:
                            existing = client.crm.contacts.basic_api.get_by_id(
                                contact_id=email,
                                id_property="email"
                            )
                            # Update existing
                            client.crm.contacts.basic_api.update(
                                contact_id=existing.id,
                                simple_public_object_input={"properties": properties}
                            )
                            stats['updated'] += 1
                        except ContactsApiException:
                            # Contact doesn't exist, create new
                            if update_existing:
                                properties['email'] = email
                                client.crm.contacts.basic_api.create(
                                    simple_public_object_input={"properties": properties}
                                )
                                stats['created'] += 1
                            else:
                                stats['failed'] += 1
                                if len(stats['errors']) < 10:
                                    stats['errors'].append(f"Row {idx}: Contact not found and create disabled")

                    time.sleep(0.1)  # Rate limiting

                except ContactsApiException as e:
                    stats['failed'] += 1
                    if len(stats['errors']) < 10:
                        stats['errors'].append(f"Row {idx}: Hubspot API error - {str(e)[:100]}")

            except (ValueError, AttributeError, KeyError, TypeError) as e:
                stats['failed'] += 1
                if len(stats['errors']) < 10:
                    stats['errors'].append(f"Row {idx}: Unexpected error - {str(e)[:100]}")

        print(f"Processed batch {batch_start//batch_size + 1}: {stats['updated'] + stats['created']} successful, {stats['failed']} failed")

    return stats

