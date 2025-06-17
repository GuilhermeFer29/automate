# Automação de Desafios em Python

Este repositório contém soluções para uma série de desafios de automação, implementados em Python, utilizando bibliotecas como Selenium, Pandas, Requests, TensorFlow/Keras, entre outras.

## Desafios Resolvidos

A seguir, uma descrição de cada desafio resolvido, categorizados por dificuldade.

### 1 - Fácil

#### 01 - Coletar Dados
*   **Localização do Script:** `desafios/1 - Fácil/01 - Coletar Dados/script.py`
*   **Descrição:** Este script utiliza Selenium para navegar no site de e-commerce de teste (`webscraper.io`). Ele acessa as categorias "Laptops", "Tablets" e "Phones", extrai informações detalhadas de cada produto (nome, preço, descrição, avaliação em estrelas e número de reviews). Os dados coletados para cada categoria são então organizados e salvos em arquivos Excel (`.xlsx`) distintos. Cada arquivo Excel contém múltiplas planilhas, com os produtos ordenados por diferentes critérios: preço, número de reviews, avaliação em estrelas, e uma ordenação bônus combinada.

#### 02 - Coletar Dados (Paginação)
*   **Localização do Script:** `desafios/1 - Fácil/02 - Coletar Dados (Paginação)/script.py`
*   **Descrição:** Este script também utiliza Selenium para interagir com o site `webscraper.io`. A principal diferença em relação ao desafio anterior é a capacidade de lidar com paginação. Para cada categoria de produto ("Laptops", "Tablets", "Phones"), o script coleta os dados dos produtos e, se houver múltiplas páginas de resultados, ele navega por todas elas clicando no botão "próximo" até que não haja mais páginas. Todos os produtos coletados de todas as páginas de uma categoria são então consolidados. Assim como no desafio anterior, os dados são salvos em arquivos Excel (`.xlsx`) com múltiplas planilhas ordenadas por preço, reviews, estrelas e uma ordenação bônus.

#### 03 - Coletar Dados (Expandir)
*   **Localização do Script:** `desafios/1 - Fácil/03 - Coletar Dados (Expandir)/script.py`
*   **Descrição:** Este script expande a coleta de dados do site `webscraper.io`. Além de navegar pelas categorias e lidar com a paginação (como no Desafio 02), ele vai um passo além: para cada produto listado, o script clica no link do produto para acessar sua página de detalhes individual. Nessa página de detalhes, ele coleta não apenas as informações básicas (nome, preço, descrição, estrelas, reviews), mas também extrai características adicionais específicas do produto que estão geralmente em uma seção de "especificações" ou "detalhes". Todos esses dados, incluindo as características expandidas, são então compilados e salvos em um único arquivo Excel (`.xlsx`) por categoria, com o nome `_expandido.xlsx`.

### 2 - Médio

#### 01 - RPA Challenge (Input)
*   **Localização do Script:** `desafios/2 - Médio/01 - RPA Challenge (Input)/script.py`
*   **Descrição:** Este script automatiza o preenchimento de um formulário no site `rpachallenge.com`. Primeiramente, ele baixa um arquivo Excel (`challenge.xlsx`) do site, que contém os dados a serem inseridos. Em seguida, para cada linha de dados no arquivo Excel, o script localiza os campos correspondentes no formulário web (First Name, Last Name, Company Name, etc.) e os preenche. Uma característica importante deste desafio é que a ordem dos campos no formulário pode mudar a cada rodada, então o script localiza os campos dinamicamente usando seus atributos `ng-reflect-name` antes de inserir os dados. Após preencher todos os campos para uma rodada, ele clica em "Submit" e repete o processo para a próxima linha do Excel. Ao final, ele captura e exibe a mensagem de resultado do desafio.

#### 02 - Agrupar Dados (Excel)
*   **Localização do Script:** `desafios/2 - Médio/02 - Agrupar Dados (Excel)/script.py`
*   **Descrição:** Este script lida com dados de licitações públicas do TCE-RS. As etapas principais são:
    1.  **Download e Extração**: Baixa um arquivo `.zip` contendo dados de licitações do ano de 2024 do portal de dados do TCE-RS. Se o arquivo já existir, pula o download. Em seguida, extrai o conteúdo do arquivo zip, que inclui três arquivos CSV principais: `licitacao.csv`, `lote.csv`, e `item.csv`.
    2.  **Processamento e Organização**:
        *   Lê os três arquivos CSV para DataFrames do Pandas.
        *   Itera sobre cada licitação no `licitacao_df`.
        *   Para cada licitação, cria uma estrutura de diretórios hierárquica em uma pasta chamada `licitacoes`. O nome de cada pasta de licitação é formado pela concatenação do código do órgão, número da licitação, ano e modalidade.
        *   Dentro de cada pasta de licitação, salva um arquivo `link.txt` contendo o link direto para a licitação no portal Licitacon Cidadão.
        *   Cria uma subpasta `lotes` dentro de cada pasta de licitação.
        *   Filtra os lotes correspondentes a cada licitação do `lote_df`.
        *   Para cada lote, filtra os itens correspondentes desse lote a partir do `item_df`.
        *   Salva os itens de cada lote em um arquivo CSV separado (nomeado como `[NR_LOTE].csv`) dentro da respectiva subpasta `lotes`.
    O script utiliza `tqdm` para exibir uma barra de progresso durante o download do arquivo.

#### 03 - Coletar Dados (Scroll)
*   **Localização do Script:** `desafios/2 - Médio/03 - Coletar Dados (Scroll)/script.py`
*   **Descrição:** Este script lida com a coleta de dados de uma página de e-commerce (`webscraper.io/test-sites/e-commerce/scroll`) que carrega mais produtos à medida que o usuário rola a página para baixo (scroll infinito).
    1.  **Navegação e Scroll**: Para cada categoria de produto ("Laptops", "Tablets", "Phones"), o script navega até a página correta. Em seguida, ele rola a página para baixo repetidamente usando JavaScript (`window.scrollTo(0, document.body.scrollHeight)`). Após cada rolagem, ele aguarda um pouco e verifica se novos produtos foram carregados comparando a altura atual da página com a altura anterior. O processo de rolagem continua até que não haja mais produtos novos carregados.
    2.  **Coleta de Dados**: Uma vez que todos os produtos foram carregados, o script coleta as informações de cada um (nome, preço, descrição, estrelas, reviews).
    3.  **Salvamento em Excel**: Os dados coletados para cada categoria são armazenados em DataFrames do Pandas. Finalmente, todos os DataFrames são salvos em um único arquivo Excel (`produtos_scroll.xlsx`). Cada categoria tem suas próprias planilhas dentro deste arquivo, com os produtos ordenados por preço, reviews, estrelas e uma ordenação bônus.

### 3 - Difícil

#### 01 - Text Captcha

Este desafio envolve o reconhecimento de caracteres em imagens CAPTCHA. Duas abordagens principais foram desenvolvidas:

##### a) Solução com Modelo Pré-treinado e OCR (Tesseract)
*   **Localização do Script:** `desafios/3 - Difícil/01 - Text Captcha/script.py`
*   **Descrição:** Este script tenta resolver CAPTCHAs do site `captcha.com/demos/features/captcha-demo.aspx`.
    1.  **Carregamento do Modelo e Selenium**: Carrega um modelo Keras (`final_model.h5`) pré-treinado e inicializa um driver Selenium.
    2.  **Loop de Tentativas**: Localiza a imagem do CAPTCHA (link direto ou base64), baixa/decodifica.
    3.  **Predição com Modelo Próprio**: A imagem é pré-processada e o modelo carregado prediz os 10 caracteres.
    4.  **Fallbacks com OCR (Tesseract)**: Se a predição do modelo falhar ou a imagem for JPEG numérica, usa `pytesseract` com whitelists específicas.
    5.  **Envio e Avaliação**: Envia o texto resolvido, verifica o resultado, calcula a taxa de acerto e continua até atingir 75% de acerto em 10 tentativas.

##### b) Sistema Avançado de Treinamento de Modelo CAPTCHA
*   **Localização do Script Principal:** `captcha_cnn/advanced_solver.py`
*   **Descrição da Solução:** Para criar um resolvedor de CAPTCHA robusto e adaptável, foi implementado um sistema completo de treinamento de um novo modelo de Rede Neural Convolucional (CNN) com as seguintes características:
    *   **Modelo CNN Multi-Output:**
        *   Utiliza uma arquitetura CNN para extração de características das imagens CAPTCHA.
        *   Possui múltiplas cabeças de saída (uma para cada caractere do CAPTCHA, configurado para `max_length=10`), permitindo a predição de cada caractere individualmente.
        *   As camadas de saída utilizam ativação linear para produzir logits, e a função de perda `SparseCategoricalCrossentropy` é configurada com `from_logits=True` para maior estabilidade numérica e eficiência.
    *   **Pipeline de Dados Flexível e Eficiente:**
        *   **Tratamento de Labels:** O sistema aceita labels de CAPTCHA com tamanho variável. Labels menores que o `max_length` (10 caracteres) são preenchidos com o caractere `_` (underscore) à direita, e labels maiores são truncados para garantir um tamanho fixo para o modelo.
        *   **Processamento Misto de Imagens:** O modelo é treinado para reconhecer imagens CAPTCHA fornecidas tanto como caminhos de arquivo quanto como strings codificadas em base64. Durante a fase de pré-processamento, uma porção das imagens é convertida aleatoriamente para o formato base64 antes de ser processada, garantindo que o modelo seja robusto a ambos os tipos de entrada.
        *   **Pré-processamento e Cache de Dados:**
            *   As imagens são convertidas para escala de cinza, redimensionadas para um tamanho padrão (40x120 pixels) e normalizadas (valores de pixel entre 0 e 1).
            *   Um sistema de cache explícito foi implementado: os dados processados (imagens como arrays NumPy e labels codificados) são salvos em arquivos `.npz` (por exemplo, `.captcha_cache/train_data_cache.npz`). Em execuções subsequentes, se esses arquivos de cache existirem, os dados são carregados diretamente deles, pulando a demorada etapa de pré-processamento.
    *   **Monitoramento e Feedback Visual Detalhado:**
        *   **Barras de Progresso em Múltiplas Etapas:**
            *   **Varredura de Arquivos:** Uma barra de progresso `tqdm` é exibida durante a varredura inicial dos diretórios em busca dos arquivos de imagem.
            *   **Pré-processamento de Dados:** Barras de progresso `tqdm` separadas e detalhadas (com contagem, velocidade e tempo estimado) são mostradas durante o pré-processamento e criação dos arquivos de cache para os conjuntos de treino e validação.
            *   **Treinamento de Épocas:** O callback `TqdmCallback(verbose=2)` do Keras é utilizado para exibir uma barra de progresso limpa e que se atualiza em tempo real para cada época de treinamento, mostrando o avanço dos lotes (batches) e as métricas de `loss` e `accuracy`.
        *   **Logging Informativo:** Mensagens de log são geradas para cada etapa principal do pipeline, facilitando o acompanhamento e a depuração.
    *   **Configuração e Otimização do Treinamento:**
        *   O modelo é treinado utilizando o otimizador Adam.
        *   São empregados callbacks essenciais do Keras: `ModelCheckpoint`, `EarlyStopping`, e `ReduceLROnPlateau`.
    *   **Como Executar o Treinamento:**
        O script de treinamento `captcha_cnn/advanced_solver.py` pode ser executado a partir da linha de comando.
        Exemplo:
        ```bash
        python captcha_cnn/advanced_solver.py --data_dir caminho/para/imagens_captcha --model_save_path caminho/para/modelo_salvo --epochs 20 --batch_size 64
        ```

---
*Este README foi gerado e atualizado por Cascade, seu assistente de codificação IA.*
