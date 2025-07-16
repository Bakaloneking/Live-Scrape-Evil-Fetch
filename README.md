# Live☆Scrape / Evil★Fetch

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Uma aplicação de desktop inteligente que utiliza web scraping e machine learning para aprender suas preferências visuais e extrair imagens de páginas da web automaticamente.

## Sobre o Projeto

A aplicação opera em duas fases, inspiradas no arquétipo de Yu-Gi-Oh! "Evil Twin":
-   **Live☆Scrape:** O modo de "coleta de dados", onde você navega por uma página e ensina ativamente a IA, mostrando a ela quais imagens lhe interessam.
-   **Evil★Fetch:** O modo de "previsão", onde a IA, já treinada, age por conta própria para "buscar" (fetch) e baixar as imagens que ela acredita que você vai gostar de qualquer nova URL, salvando-as em pastas com o título da página.

Com uma interface gráfica simples construída em Tkinter e Pillow, a aplicação permite que o usuário gerencie seu próprio dataset de treinamento, treine o modelo de IA e o utilize para automatizar a curadoria de imagens da web.

## Estrutura do Projeto

```
/
|-- data/
|   |-- image_dataset_features.csv  (Gerado pela aplicação)
|-- models/
|   |-- image_model.joblib          (Gerado pela aplicação)
|-- extrator_ia.py                  (Lógica principal e de linha de comando)
|-- app_gui.py                      (Aplicação com interface gráfica)
|-- requirements.txt                (Bibliotecas necessárias)
|-- .gitignore
|-- LICENSE
|-- README.md
```

## Instalação e Configuração

**Pré-requisitos:**
-   Python 3.8 ou superior
-   pip (gerenciador de pacotes do Python)

**Passos:**

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Instale as dependências:**
    Recomenda-se criar um ambiente virtual (`venv`) primeiro.
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

A aplicação pode ser usada através da interface gráfica, que é a forma recomendada.

**Execute a aplicação GUI:**
```bash
python app_gui.py
```

Dentro da aplicação, você terá três ações principais:

### 1. Modo Coleta (Live☆Scrape)
-   Insira a URL da página da qual deseja coletar dados.
-   Clique em **"Coletar Dados"**.
-   Uma nova janela aparecerá mostrando as imagens encontradas. Marque as que você gosta.
-   Clique em "Confirmar Seleção e Salvar". Os dados serão salvos no arquivo `data/image_dataset_features.csv`.
-   **Repita este processo para vários sites** para construir um dataset rico e variado.

### 2. Modo Treinamento
-   Clique no botão **"Treinar Modelo"**.
-   A aplicação usará todos os dados do arquivo `data/image_dataset_features.csv` para treinar a IA.
-   O modelo treinado será salvo como `models/image_model.joblib`.

### 3. Modo Previsão (Evil★Fetch)
-   Certifique-se de que já treinou um modelo.
-   Insira a URL da página que você deseja analisar.
-   Clique em **"Prever & Baixar"**.
-   A IA usará o modelo salvo para prever quais imagens você gostará.
-   As imagens selecionadas serão baixadas para uma nova pasta com o nome do título da página.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
