from langchain_core.tools import tool

import pandas as pd
import os
import requests
from dotenv import load_dotenv
load_dotenv()

AEMET_API_KEY = os.getenv('AEMET_API_KEY')


@tool("location_tool", description="A tool to get the current location of the user")
def location_tool() -> str:
    """
    This tool uses the ip-api.com API to get the current location of the user based on their IP address.
    Returns:
        - A string with the current location of the user in the format "city, country".
    """
    response = requests.get('http://ip-api.com/json/')
    data = response.json()

    return f"The current location of the user is {data['city']}, {data['country']}."


@tool("get_province_code", description="A tool to get the province code for a given province name")
def get_province_code(province_name: str) -> str:
    """
    This tool uses a CSV file with the province names and codes to get the province code for a given province name.
    Args:
        - province_name: The name of the province to get the code for.
    Returns:
        - A string with the province code or an error message if the province name is not found
    """
    df = pd.read_csv("data/dict_municipios.csv", dtype=str)
    province_code = df[df['NOMBRE'] == province_name]['CPRO'].unique()

    if len(province_code) == 0:
        return f'Error, no existe ninguna provincia llamada {province_name}'
    if len(province_code) == 1:
        return f'El código de la provincia {province_name} es {province_code[0]}'
    
    return f"Hay varios códigos para la provincia {province_name}: {', '.join(code[0] for code in province_code)}"


@tool("get_weather", description=" A tool to get the weather of a province code")
def get_weather(province_code: str) -> str:
    """
    This tool uses the AEMET Open Data API to get the weather of a province code.
    Args:
        - province_code: The code of the province to get the weather information for.
    Returns:
        - A string with the weather information or an error message if the API request fails.
    """
    try:
        response = requests.get(
            url=f"https://opendata.aemet.es/opendata/api/prediccion/provincia/hoy/{province_code}/?api_key={AEMET_API_KEY}"
        )
        response.raise_for_status()

        response = response.json()

        if response.get("estado") != 200:
            raise ValueError(f"Error en la petición AEMET Open Data: {response.get('descripcion', 'Sin descripción')}")

        url_data = response.get("datos")
        url_metadata = response.get("metadatos")
    except Exception as e:
        return e
    
    try:
        response = requests.get(url_data)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return e
