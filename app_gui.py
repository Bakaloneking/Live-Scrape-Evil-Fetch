# -*- coding: utf-8 -*-
# Arquivo: app_gui.py

# Importações Padrão
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import csv
import io
from PIL import Image, ImageTk
import pandas as pd

# Importa as funções do nosso outro arquivo
# (Assumimos que extrator_ia.py está na mesma pasta que app_gui.py)
from extrator_ia import treinar_modelo, prever_e_baixar, extract_features

# ==============================================================================
# SEÇÃO DE CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS
# ==============================================================================
# Descobre o caminho absoluto do diretório onde este script (app_gui.py) está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define os nomes das subpastas para organização
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Cria os caminhos completos para os arquivos
ARQUIVO_DATASET = os.path.join(DATA_DIR, "image_dataset_features.csv")
ARQUIVO_MODELO = os.path.join(MODELS_DIR, "image_model.joblib")

# Cria as subpastas 'data' e 'models' se elas não existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================


# --- Classe da Janela de Seleção (sem alterações) ---
class SelectionWindow(tk.Toplevel):
    # ... (cole aqui a classe SelectionWindow completa da resposta anterior)
    def __init__(self, parent, image_list, callback):
        super().__init__(parent)
        self.title("Selecione as Imagens para o Dataset")
        self.geometry("800x600")

        self.image_list = image_list
        self.callback = callback
        self.check_vars = []

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for i, img_info in enumerate(self.image_list):
            item_frame = ttk.Frame(scrollable_frame, borderwidth=1, relief="solid", padding=5)
            item_frame.pack(fill="x", padx=10, pady=5)

            var = tk.BooleanVar()
            self.check_vars.append(var)
            cb = ttk.Checkbutton(item_frame, text=f"[{i + 1}]", variable=var)
            cb.pack(side="left", anchor="n", padx=10)

            if img_info.get('thumbnail'):
                img_label = ttk.Label(item_frame, image=img_info['thumbnail'])
                img_label.pack(side="right", padx=10)

            width = img_info.get('real_width') or img_info.get('width', 'N/A')
            height = img_info.get('real_height') or img_info.get('height', 'N/A')

            info_text = (f"URL: {img_info['url']}\nDimensões: {width} x {height}\n"
                         f"Formato: {img_info.get('format', 'N/A')}\nAlt Text: {img_info.get('alt', 'N/A')}")
            info_label = ttk.Label(item_frame, text=info_text, wraplength=550, justify="left")
            info_label.pack(side="left", fill="x", expand=True)

        confirm_button = ttk.Button(self, text="Confirmar Seleção e Salvar", command=self.confirm_and_save)
        confirm_button.pack(pady=10)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.transient(parent)
        self.grab_set()

    def confirm_and_save(self):
        labeled_data = []
        for i, img_info in enumerate(self.image_list):
            label = 1 if self.check_vars[i].get() else 0
            img_info['selected'] = label
            img_info.pop('thumbnail', None)
            labeled_data.append(img_info)
        self.callback(labeled_data)
        self.destroy()


# --- Classe Principal da Aplicação ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # ... (cole aqui o método __init__ completo da resposta anterior)
        self.title("Extrator de Imagens com IA")
        self.geometry("700x500")
        self.frame = ttk.Frame(self, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.frame, text="URL da Página Web:").pack(anchor="w")
        self.url_entry = ttk.Entry(self.frame, width=80)
        self.url_entry.pack(fill=tk.X, pady=5)
        self.url_entry.insert(0, "Cole a URL aqui")
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=10)
        self.btn_coletar = ttk.Button(button_frame, text="Coletar Dados", command=self.iniciar_coleta)
        self.btn_coletar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.btn_treinar = ttk.Button(button_frame, text="Treinar Modelo", command=self.iniciar_treino)
        self.btn_treinar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.btn_prever = ttk.Button(button_frame, text="Prever & Baixar", command=self.iniciar_previsao)
        self.btn_prever.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Label(self.frame, text="Status:").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, height=15)
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_area.configure(state='disabled')

    def log(self, message):
        # ... (função log sem alterações)
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.configure(state='disabled')
        self.log_area.see(tk.END)

    def run_task_in_thread(self, task_function, *args):
        # ... (função run_task_in_thread sem alterações)
        thread = threading.Thread(target=task_function, args=args)
        thread.start()

    def iniciar_coleta(self):
        # ... (cole aqui a função iniciar_coleta completa da resposta anterior)
        url = self.url_entry.get()
        if "Cole a URL aqui" in url or not url:
            messagebox.showerror("Erro", "Por favor, insira uma URL válida.")
            return

        self.log(f"--- Buscando e processando imagens em: {url} (Isso pode demorar)... ---")

        def scrape_and_process_images():
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')

                features_list = []
                image_tags = soup.find_all('img')

                for i, img_tag in enumerate(image_tags):
                    self.log(f"Processando imagem {i + 1}/{len(image_tags)}...")
                    features = extract_features(img_tag, resp.url)
                    if not features: continue

                    try:
                        img_resp = requests.get(features['url'], timeout=5, stream=True)
                        img_resp.raise_for_status()
                        img_data = img_resp.raw.read()
                        img_obj = Image.open(io.BytesIO(img_data))
                        features['real_width'] = img_obj.width
                        features['real_height'] = img_obj.height
                        features['format'] = img_obj.format
                        img_obj.thumbnail((100, 100))
                        features['thumbnail'] = ImageTk.PhotoImage(img_obj)
                    except Exception:
                        features['thumbnail'] = None

                    features_list.append(features)

                if not features_list:
                    self.log("Nenhuma imagem processável foi encontrada.")
                    return

                self.log("--- Processamento concluído. Abrindo janela de seleção... ---")
                self.after(0, lambda: SelectionWindow(self, features_list, self.salvar_dados_coletados))

            except Exception as e:
                self.log(f"Erro ao buscar imagens: {e}")

        self.run_task_in_thread(scrape_and_process_images)

    def salvar_dados_coletados(self, labeled_data):
        """Função chamada pela janela de seleção para salvar os dados no CSV."""
        # ALTERAÇÃO: Usa a variável global ARQUIVO_DATASET
        self.log(f"Salvando {len(labeled_data)} entradas no arquivo '{ARQUIVO_DATASET}'...")

        colunas_csv = [
            'url', 'alt', 'width', 'height', 'extension', 'parent_tag',
            'real_width', 'real_height', 'format', 'selected'
        ]

        df_novos_dados = pd.DataFrame(labeled_data)

        # ALTERAÇÃO: Usa a variável global ARQUIVO_DATASET
        if os.path.exists(ARQUIVO_DATASET):
            df_existente = pd.read_csv(ARQUIVO_DATASET)
            df_final = pd.concat([df_existente, df_novos_dados], ignore_index=True)
            df_final.drop_duplicates(subset=['url'], keep='last', inplace=True)
        else:
            df_final = df_novos_dados

        for col in colunas_csv:
            if col not in df_final.columns:
                df_final[col] = None

        df_final.to_csv(ARQUIVO_DATASET, index=False, columns=colunas_csv)
        self.log(f"--- Sucesso! Dados salvos. Total de {len(df_final)} entradas no dataset. ---")

    def task_wrapper(self, task_function, *args):
        """Redireciona a saida 'print' das funções para a área de log da GUI."""
        import sys
        class IORedirector:
            def __init__(self, widget): self.widget = widget

            def write(self, str): self.widget.log(str.strip())

            def flush(self): pass

        original_stdout = sys.stdout
        sys.stdout = IORedirector(self)
        try:
            task_function(*args)
        finally:
            sys.stdout = original_stdout
        self.log("--- TAREFA CONCLUÍDA ---")

    def iniciar_treino(self):
        """Função do botão para iniciar o treinamento."""
        self.log("--- Iniciando Treinamento do Modelo... ---")
        # ALTERAÇÃO: Passa os caminhos absolutos para a função de treino
        self.run_task_in_thread(self.task_wrapper, treinar_modelo, ARQUIVO_DATASET, ARQUIVO_MODELO)

    def iniciar_previsao(self):
        """Função do botão para iniciar a previsão."""
        url = self.url_entry.get()
        if "Cole a URL aqui" in url or not url:
            messagebox.showerror("Erro", "Por favor, insira uma URL válida.")
            return
        self.log(f"--- Iniciando Previsão da URL: {url} ---")
        # ALTERAÇÃO: Passa a URL e o caminho absoluto do modelo para a função de previsão
        self.run_task_in_thread(self.task_wrapper, prever_e_baixar, url, ARQUIVO_MODELO, BASE_DIR)

# --- Ponto de Entrada da Aplicação ---
if __name__ == "__main__":
    app = App()
    app.mainloop()