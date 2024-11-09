import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
import plotly.express as px
import httpx
import json
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Configuration Section ===
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),  # Load API key from environment variable
    "openai_endpoint": "https://api.openai.com/v1/chat/completions",
    "file_path": 'https://github.com/rmejia41/open_datasets/raw/main/StateHighestAnnualAverageFluoride.csv',
    "timeout": 30.0,
    "retry_attempts": 3
}

# Check if API key is loaded
if config["openai_api_key"] is None:
    raise EnvironmentError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")

# === Initialize Data and Logging ===
logging.basicConfig(level=logging.INFO)
fluoride_data = pd.read_csv(config['file_path'])

# Mapping and data preparation
state_abbreviations = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}

fluoride_data['State Name'] = fluoride_data['State'].map(state_abbreviations)
fluoride_data['Highest Adjusted CWS Monthly Fluoride Average'] = fluoride_data[
    'Highest Adjusted CWS Monthly Fluoride Average'].fillna(0)

# State-specific guidelines
state_guidelines = {
    "Alabama": "https://www.alabamapublichealth.gov/oralhealth/fluoridation.html",
    "Alaska": "http://dhss.alaska.gov/dph/wcfh/Pages/oralhealth/fluoride.aspx",
    "Arizona": "https://www.azdhs.gov/prevention/womens-childrens-health/oral-health/index.php#community-water-fluoridation",
    "Arkansas": "https://www.healthy.arkansas.gov/programs-services/topics/community-water-fluoridation",
    "California": "https://www.cdph.ca.gov/Programs/CCDPHP/DCDIC/CDCB/Pages/OralHealthProgram/Fluoridation.aspx",
    "Colorado": "https://cdphe.colorado.gov/fluoride",
    "Connecticut": "https://portal.ct.gov/DPH/Health-Education-Management--Surveillance/Oral-Health/Community-Water-Fluoridation",
    "Delaware": "https://www.dhss.delaware.gov/dhss/dph/hsm/fluoridation.html",
    "Florida": "http://www.floridahealth.gov/programs-and-services/community-health/dental-health/fluoridation/index.html",
    "Georgia": "https://dph.georgia.gov/oral-health/fluoridation",
    "Hawaii": "https://health.hawaii.gov/about/contact/",
    "Idaho": "https://healthandwelfare.idaho.gov/",
    "Illinois": "http://www.dph.illinois.gov/",
    "Indiana": "https://www.in.gov/health/",
    "Iowa": "https://idph.iowa.gov/",
    "Kansas": "http://www.kdheks.gov/",
    "Kentucky": "https://chfs.ky.gov/",
    "Louisiana": "https://ldh.la.gov/",
    "Maine": "https://www.maine.gov/dhhs/",
    "Maryland": "https://health.maryland.gov/",
    "Massachusetts": "https://www.mass.gov/orgs/department-of-public-health",
    "Michigan": "https://www.michigan.gov/mdhhs/adult-child-serv/childrenfamilies/familyhealth/oralhealth/community-water-fluoridation",
    "Minnesota": "https://www.health.state.mn.us/",
    "Mississippi": "https://msdh.ms.gov/",
    "Missouri": "https://health.mo.gov/",
    "Montana": "https://dphhs.mt.gov/",
    "Nebraska": "http://dhhs.ne.gov/",
    "Nevada": "https://dpbh.nv.gov/",
    "New Hampshire": "https://www.dhhs.nh.gov/",
    "New Jersey": "https://www.nj.gov/health/",
    "New Mexico": "https://www.nmhealth.org/",
    "New York": "https://www.health.ny.gov/prevention/dental/fluoridation/",
    "North Carolina": "https://www.ncdhhs.gov/",
    "North Dakota": "https://www.health.nd.gov/",
    "Ohio": "https://odh.ohio.gov/",
    "Oklahoma": "https://oklahoma.gov/health.html",
    "Oregon": "https://www.oregon.gov/oha/",
    "Pennsylvania": "https://www.health.pa.gov/",
    "Rhode Island": "https://health.ri.gov/",
    "South Carolina": "https://www.scdhec.gov/",
    "South Dakota": "https://doh.sd.gov/",
    "Tennessee": "https://www.tn.gov/health.html",
    "Texas": "https://www.dshs.state.tx.us/dental/oralhealthfluoridation.shtm",
    "Utah": "https://health.utah.gov/",
    "Vermont": "https://www.healthvermont.gov/wellness/oral-health/fluoride",
    "Virginia": "https://www.vdh.virginia.gov/",
    "Washington": "https://www.doh.wa.gov/",
    "West Virginia": "https://dhhr.wv.gov/",
    "Wisconsin": "https://www.dhs.wisconsin.gov/oral-health/index.htm",
    "Wyoming": "https://health.wyo.gov/publichealth/mch/oralhealth/",
    "District of Columbia": "https://dchealth.dc.gov/"
}


# === AI Feedback Handler Class ===
class AIFeedbackGenerator:
    def __init__(self, api_key, endpoint, timeout=30, retries=3):
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_feedback(self, state_name, year, max_fluoride, avg_fluoride, cws_name):
        guideline_link = state_guidelines.get(state_name, None)
        if guideline_link:
            additional_info = f"For specific state guidelines, refer to: {guideline_link}."
        else:
            additional_info = (
                "For general information on fluoride levels, refer to the CDC’s WFRS (Water Fluoridation Reporting System) or "
                "the 'My Water Fluoride' tool available through the CDC website."
            )

        prompt = (
            f"Provide a summary of the water fluoridation data for '{cws_name}' in {state_name} for the year {year}."
            f" This CWS had the highest monthly adjusted fluoride level, recorded at {max_fluoride:.2f} mg/L, with a historical average of {avg_fluoride:.2f} ppm."
            f" Please indicate if the data suggests consistency with public health guidelines or if it indicates a need for intervention. {additional_info}"
        )

        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are an assistant for public health data summaries."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 350
        }

        for attempt in range(self.retries):
            try:
                with httpx.Client(verify=False, timeout=self.timeout) as client:
                    response = client.post(self.endpoint, headers=self.headers, data=json.dumps(payload))
                response.raise_for_status()
                completion = response.json()["choices"][0]["message"]["content"].strip()
                return completion
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logging.error("Unauthorized access - check your API key.")
                    return "Error: Unauthorized access. Please check your API key."
                else:
                    logging.error(f"HTTP error on attempt {attempt + 1}: {e}")
            except httpx.RequestError as e:
                logging.error(f"Request error on attempt {attempt + 1}: {e}")
            except httpx.TimeoutException:
                logging.error(f"Timeout on attempt {attempt + 1}")

        return "Feedback generation failed after multiple attempts."


# Initialize the AI feedback generator with API key
ai_feedback_generator = AIFeedbackGenerator(
    api_key=config["openai_api_key"],
    endpoint=config["openai_endpoint"],
    timeout=config["timeout"],
    retries=config["retry_attempts"]
)

# === Dash App Initialization ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server=app.server

# === App Layout ===
app.layout = dbc.Container([
    html.H1("State Highest Adjusted Fluoride Average: CWS Data", className="text-center my-4"),

    dbc.Row([
        dbc.Col([html.Label("Select Year"), dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': year, 'value': year} for year in fluoride_data['Year'].unique()],
            value=None, clearable=True
        )], width=6),
        dbc.Col([html.Label("Select State"), dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': state_abbreviations[state], 'value': state} for state in
                     fluoride_data['State'].unique()],
            value=None, clearable=True
        )], width=6)
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(id='map', figure=px.choropleth(scope="usa"), style={'height': '40vh'})], width=6),
        dbc.Col([dash_table.DataTable(
            id='data-table',
            columns=[
                {"name": "State", "id": "State Name"},
                {"name": "Year", "id": "Year"},
                {"name": "CWS Adjusted Name", "id": "CWS Adjusted Name"},
                {"name": "Highest Adjusted CWS Monthly Fluoride Average",
                 "id": "Highest Adjusted CWS Monthly Fluoride Average"}
            ],
            sort_action="native",
            style_table={'height': '40vh', 'overflowY': 'auto', 'width': '100%'}
        )], width=6)
    ]),

    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H5("AI-Generated Feedback", className="text-center mb-3"),
            dcc.Loading(
                id="loading-feedback",
                type="circle",
                children=html.Div(
                    id='ai-feedback',
                    style={'padding': '10px', 'background-color': '#f9f9f9'}
                )
            )
        ])])], width=12)
    ])
], fluid=True)


# === Callback ===
@app.callback(
    [Output('map', 'figure'), Output('data-table', 'data'), Output('ai-feedback', 'children')],
    [Input('year-dropdown', 'value'), Input('state-dropdown', 'value')]
)
def update_charts(selected_year, selected_state):
    if not selected_year or not selected_state:
        return px.choropleth(scope="usa"), [], "Select a year and state to view data."

    state_name = state_abbreviations[selected_state]
    state_data_all_years = fluoride_data[fluoride_data['State'] == selected_state]
    state_data_for_year = state_data_all_years[state_data_all_years['Year'] == selected_year]

    fig_map = px.choropleth(
        state_data_for_year, locations="State", locationmode="USA-states",
        color="Highest Adjusted CWS Monthly Fluoride Average",
        hover_name="State Name", scope="usa"
    ).update_layout(height=450, geo=dict(projection_scale=1.2), showlegend=False)

    table_data = state_data_all_years.to_dict('records')

    if state_data_for_year.empty:
        feedback = f"No data for {state_name} in {selected_year}."
    else:
        avg_fluoride = state_data_all_years['Highest Adjusted CWS Monthly Fluoride Average'].mean()
        max_fluoride = state_data_for_year['Highest Adjusted CWS Monthly Fluoride Average'].max()
        cws_name = state_data_for_year['CWS Adjusted Name'].iloc[0]

        feedback = ai_feedback_generator.generate_feedback(state_name, selected_year, max_fluoride, avg_fluoride,
                                                           cws_name)

    return fig_map, table_data, feedback

if __name__ == '__main__':
    app.run(debug=False)
