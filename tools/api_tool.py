from langchain_core.tools import tool

import pandas as pd
import requests


@tool("location_tool", description="A tool to get the current location of the user")
def location_tool() -> str:
    response = requests.get('http://ip-api.com/json/')
    data = response.json()

    return f"The current location of the user is {data['city']}, {data['country']}."


@tool("get_province_code", description="A tool to get the province code for a given province name")
def get_province_code(province_name: str) -> str:
    df = pd.read_csv("../data/dict_municipios.csv", dtype=str)
    province_code = df[df['NOMBRE'] == province_name]['CPRO'].unique()

    if len(province_code) == 0:
        return f'Error, no existe ninguna provincia llamada {province_name}'
    if len(province_code) == 1:
        return f'El código de la provincia {province_name} es {province_code[0]}'
    
    return f'Hay varios códigos para la provincia {province_name}: {', '.join(code[0] for code in province_code)}'


@tool("get_weather", description=" A tool to get the weather of a province code")
def get_weather(province_code: str) -> str:
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
