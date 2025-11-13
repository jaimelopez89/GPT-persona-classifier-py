# Hubspot Integration Setup Guide

This guide explains how to set up and use the Hubspot integration for pulling report data and importing classified personas.

## Prerequisites

1. **Install the Hubspot API client:**
   ```bash
   pip install hubspot-api-client
   ```

2. **Get your Hubspot Access Token:**
   - Go to https://app.hubspot.com/private-apps
   - Create a new private app
   - Grant the following scopes:
     - `crm.objects.contacts.read` - To read contacts
     - `crm.objects.contacts.write` - To update/create contacts
     - `reports.read` - To read reports (if using report-based pulling)
   - Copy the access token

3. **Set the environment variable:**
   Add to your `.env` file:
   ```
   HUBSPOT_ACCESS_TOKEN=your_access_token_here
   ```

## Creating Custom Properties in Hubspot

Before importing classified personas, you need to create custom properties in Hubspot:

1. Go to Hubspot Settings → Properties → Contact properties
2. Create the following custom properties:
   - **Property name:** `persona`
     - Type: Single-line text
     - Label: Persona
   - **Property name:** `persona_certainty`
     - Type: Single-line text (or Number if you prefer)
     - Label: Persona Certainty

## Finding Your Report ID

To pull data from a specific Hubspot report:

1. Open the report in Hubspot
2. Look at the URL - it will contain the report ID
   - Example: `https://app.hubspot.com/reports/12345678/...`
   - The number `12345678` is your report ID

## Usage

### Option 1: Pull from Hubspot Report (Automatic)

Set the report ID in `config.py`:
```python
HUBSPOT_REPORT_ID = "12345678"  # Your report ID
```

Then run without `--input`:
```bash
python streaming_enricher.py --hubspot-import
```

Or override the report ID via command line:
```bash
python streaming_enricher.py --hubspot-report 12345678 --hubspot-import
```

### Option 2: Pull from Hubspot Report (Command Line)

```bash
python streaming_enricher.py --hubspot-report 12345678 --hubspot-import
```

### Option 3: Use File Input, Import to Hubspot

```bash
python streaming_enricher.py --input /path/to/file.csv --hubspot-import
```

### Option 4: Use Hubspot Zip File, Import to Hubspot

```bash
python streaming_enricher.py --input /path/to/hubspot-export.zip --hubspot-import
```

## What Gets Imported

The import function will:
- **Exclude:** "Prospect Id" and "Skip Reason" columns
- **Include:** All other columns from the classified output
- **Map columns:**
  - "Persona" → `persona` property
  - "Persona Certainty" → `persona_certainty` property
  - "Email" → `email` property
  - "First Name" → `firstname` property
  - "Last Name" → `lastname` property
  - "Job Title" → `jobtitle` property
  - "Company" → `company` property
  - Other columns → Lowercase with underscores (e.g., "My Column" → `my_column`)

## Notes

- Contacts are identified by "Prospect Id" (Hubspot contact ID) or "Email"
- If a contact exists, it will be updated
- If a contact doesn't exist and `update_existing=True`, a new contact will be created
- The import processes contacts in batches of 100 to respect rate limits
- Progress is shown during import with statistics on updated/created/failed contacts

