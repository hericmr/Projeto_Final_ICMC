

import arrow
import requests

# Lista de chaves de API que o programa tentará em sequência
api_keys = [
    "ea996e2a-88a5-11ef-ae24-0242ac130003-ea996e98-88a5-11ef-ae24-0242ac130003",
    "6b7ca118-da20-11ee-8a07-0242ac130002-6b7ca186-da20-11ee-8a07-0242ac130002",
    "1d6c47a6-8a21-11ef-a0d5-0242ac130003-1d6c4814-8a21-11ef-a0d5-0242ac130003",
    "bf5e8542-8a21-11ef-8d8d-0242ac130003-bf5e8614-8a21-11ef-8d8d-0242ac130003"
]

lat = -23.9608
lon = -46.3336

def obter_dados():
    start = arrow.now().floor('day')
    end = arrow.now().shift(days=1).floor('day')

    for api_key in api_keys:
        print(f"Tentando com a chave de API: {api_key}")
        resposta = requests.get(
            'https://api.stormglass.io/v2/astronomy/point',
            params={
                'lat': lat,
                'lng': lon,
                'start': start.to('UTC').timestamp(),
                'end': end.to('UTC').timestamp(),
            },
            headers={
                'Authorization': api_key
            }
        )

        if resposta.status_code == 200:
            # Se a requisição for bem-sucedida, exibe o JSON retornado
            dados_json = resposta.json()
            print("Dados obtidos com sucesso!")
            return dados_json
        else:
            # Se a chave falhar, exibe o erro e tenta a próxima chave
            print(f"Erro {resposta.status_code} na chave {api_key}. Tentando próxima chave...\n")

    # Se nenhuma chave funcionar, retorna None
    print("Nenhuma chave de API funcionou.")
    return None

def imprimir_dados_json(dados_json):
    if dados_json:
        # Imprime o JSON completo formatado
        import json
        print(json.dumps(dados_json, indent=4))
    else:
        print("Nenhum dado disponível para exibir.")

# Função principal que obtém e exibe os dados
def checar_condicoes_de_mergulho():
    agora = arrow.now().format('YYYY-MM-DD HH:mm:ss')
    print(f"Data e hora da verificação: {agora}")

    dados_json = obter_dados()

    if dados_json:
        # Exibe todas as informações do JSON
        imprimir_dados_json(dados_json)
    else:
        print("Falha ao obter dados da API.")

checar_condicoes_de_mergulho()

