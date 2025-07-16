# -*- coding: utf-8 -*-
# Arquivo: extrator_ia.py

# ==============================================================================
# PASSO 1: IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import shutil
import urllib.parse
import re
import joblib
import argparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# ==============================================================================
# PASSO 2: DEFINIÇÃO DAS FUNÇÕES AUXILIARES
# ==============================================================================

def extract_features(img_tag, base_url):
    """Extrai características (features) de uma única tag <img>."""
    features = {}
    src = img_tag.get('data-original') or img_tag.get('src')
    if not src: return None

    absolute_url = urllib.parse.urljoin(base_url, src)
    features.update({
        'url': absolute_url,
        'extension': os.path.splitext(urllib.parse.urlparse(absolute_url).path)[1].lower(),
        'alt': img_tag.get('alt', ''),
        'width': img_tag.get('width'),
        'height': img_tag.get('height'),
        'parent_tag': img_tag.parent.name if img_tag.parent else None
    })
    return features


def clean_dataframe(df):
    """Limpa o DataFrame para o processamento do modelo."""
    for col in ['width', 'height']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('px', '', regex=False), errors='coerce')
    text_cols = ['url', 'alt']
    for col in text_cols:
        df[col] = df[col].fillna('')
    return df


# ==============================================================================
# PASSO 3: FUNÇÕES PRINCIPAIS DA APLICAÇÃO (MODOS)
# ==============================================================================

def coletar_dados(url, arquivo_saida='image_dataset_features.csv'):
    """Modo de Coleta: Raspa uma URL, pede a seleção do usuário e salva no CSV."""
    print(f"--- Modo de Coleta de Dados: {url} ---")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        features_list = [extract_features(img, resp.url) for img in soup.find_all('img') if
                         extract_features(img, resp.url)]

        if not features_list:
            print("Nenhuma imagem encontrada na página.")
            return

        print("\n--- Por favor, selecione as imagens que você quer ---")
        for i, features in enumerate(features_list):
            print(f"[{i + 1}] URL: {features['url']}")

        user_input = input("\nDigite os números das imagens, separados por vírgula: ")
        selected_numbers = {int(num.strip()) for num in user_input.split(',')}

        for i, features in enumerate(features_list):
            features['selected'] = 1 if (i + 1) in selected_numbers else 0

        df_novos_dados = pd.DataFrame(features_list)

        if os.path.exists(arquivo_saida):
            df_existente = pd.read_csv(arquivo_saida)
            df_final = pd.concat([df_existente, df_novos_dados], ignore_index=True)
            df_final.drop_duplicates(subset=['url'], keep='last', inplace=True)
        else:
            df_final = df_novos_dados

        df_final.to_csv(arquivo_saida, index=False)
        print(f"\n--- Sucesso! Dados salvos em '{arquivo_saida}'. Total de {len(df_final)} entradas. ---")

    except Exception as e:
        print(f"Ocorreu um erro na coleta de dados: {e}")


def treinar_modelo(dataset_path='image_dataset_features.csv', model_path='image_model.joblib'):
    """Modo de Treinamento: Lê o CSV, treina o modelo de IA e o salva."""
    print(f"--- Modo de Treinamento: Usando '{dataset_path}' ---")
    try:
        df = pd.read_csv(dataset_path)
        df = clean_dataframe(df)

        if df['selected'].nunique() < 2:
            print("Erro: O dataset precisa conter exemplos de imagens selecionadas (1) e não selecionadas (0).")
            return

        y = df['selected']
        X = df.drop('selected', axis=1)

        numerical_features = ['width', 'height']
        categorical_features = ['extension', 'parent_tag']

        # --- ESTA É A CORREÇÃO ---
        # A definição completa do preprocessor, sem placeholders.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('url_text', TfidfVectorizer(max_features=100), 'url'),
                ('alt_text', TfidfVectorizer(max_features=50), 'alt')
            ],
            remainder='drop'
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        model.fit(X, y)
        joblib.dump(model, model_path)
        print(f"--- Sucesso! Modelo treinado e salvo como '{model_path}' ---")

    except FileNotFoundError:
        print(f"Erro: Arquivo de dataset '{dataset_path}' não encontrado. Execute o modo de coleta primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro no treinamento: {e}")


def prever_e_baixar(url, model_path='image_model.joblib', base_save_path='.'):
    """
    Modo de Previsão: Raspa uma URL, usa o modelo para prever e baixa as imagens.
    ALTERAÇÃO: Adicionado 'base_save_path' para definir onde salvar a pasta.
    """
    print(f"--- Modo de Previsão: {url} ---")
    try:
        loaded_model = joblib.load(model_path)

        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        page_title = soup.title.string if soup.title else 'Pagina_Sem_Titulo'
        sanitized_title = re.sub(r'[\\/*?:"<>|]', '', page_title).strip().replace(' ', '_')

        # --- ALTERAÇÃO AQUI ---
        # Cria o caminho completo para a pasta de salvamento, usando o caminho base
        save_dir = os.path.join(base_save_path, sanitized_title)

        features_list = [extract_features(img, resp.url) for img in soup.find_all('img') if
                         extract_features(img, resp.url)]

        if not features_list:
            print("Nenhuma imagem encontrada para prever.")
            return

        new_df = pd.DataFrame(features_list)
        new_df = clean_dataframe(new_df)

        predictions = loaded_model.predict(new_df)
        selected_images = new_df[predictions == 1]

        print(f"\nO modelo previu que você vai gostar de {len(selected_images)} imagem(ns).")

        if not selected_images.empty:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Baixando imagens para a pasta '{save_dir}'...")

            for index, row in selected_images.iterrows():
                img_url = row['url']
                try:
                    img_resp = requests.get(img_url, stream=True, timeout=10)
                    filename = os.path.basename(urllib.parse.urlparse(img_url).path) or f"image_{index}.jpg"
                    filepath = os.path.join(save_dir, filename)  # Agora usa o caminho completo
                    with open(filepath, 'wb') as f:
                        shutil.copyfileobj(img_resp.raw, f)
                    print(f"  - Salvo {filename}")
                except Exception as e:
                    print(f"  - Falha ao baixar {img_url}: {e}")
            print("--- Download completo ---")

    except FileNotFoundError:
        print(f"Erro: Modelo '{model_path}' não encontrado. Execute o modo de treinamento primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro na previsão: {e}")


# ==============================================================================
# PASSO 4: LÓGICA PRINCIPAL PARA EXECUTAR VIA LINHA DE COMANDO
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IA para extrair e baixar imagens de websites.")
    parser.add_argument("--coletar", type=str, help="URL para coletar novos dados de treinamento.")
    parser.add_argument("--treinar", action="store_true", help="Treina o modelo com os dados existentes.")
    parser.add_argument("--prever", type=str, help="URL para prever e baixar imagens.")
    parser.add_argument("--dataset", default="image_dataset_features.csv", help="Caminho para o arquivo do dataset CSV.")
    parser.add_argument("--modelo", default="image_model.joblib", help="Caminho para o arquivo do modelo .joblib.")
    args = parser.parse_args()

    if args.coletar:
        coletar_dados(args.coletar, arquivo_saida=args.dataset)
    elif args.treinar:
        treinar_modelo(dataset_path=args.dataset, model_path=args.modelo)
    elif args.prever:
        # --- ALTERAÇÃO AQUI ---
        # Define o caminho base como o diretório atual ao rodar via terminal
        prever_e_baixar(args.prever, model_path=args.modelo, base_save_path=os.getcwd())
    else:
        print("Nenhum modo selecionado. Use --coletar, --treinar ou --prever.")
        parser.print_help()