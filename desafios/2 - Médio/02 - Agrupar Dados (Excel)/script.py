
import os
import requests
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile

class Desafio02:
    def iniciar(self):
        url = 'https://dados.tce.rs.gov.br/dados/licitacon/licitacao/ano/2024.csv.zip'
        zip_file = 'downloads/licitacoes-consolidado-2024.csv.zip'
        extract_dir = 'downloads/extracted'
        os.makedirs('downloads', exist_ok=True)
        os.makedirs(extract_dir, exist_ok=True)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        if not os.path.exists(zip_file) or os.path.getsize(zip_file) == 0:
            r = requests.get(url, headers=headers, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            
            if r.status_code == 200:
                with open(zip_file, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc='Baixando') as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                print(f"Erro no download: {r.status_code}")
                return
        else:
            print("Arquivo já baixado. Prosseguindo com a extração.")

        with ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        licitacao_df = pd.read_csv(os.path.join(extract_dir, 'licitacao.csv'), low_memory=False)
        lote_df = pd.read_csv(os.path.join(extract_dir, 'lote.csv'), low_memory=False)
        item_df = pd.read_csv(os.path.join(extract_dir, 'item.csv'), low_memory=False)

        os.makedirs('licitacoes', exist_ok=True)

        for _, licitacao in licitacao_df.iterrows():
            org = licitacao['CD_ORGAO']
            num = licitacao['NR_LICITACAO']
            ano = licitacao['ANO_LICITACAO']
            mod = licitacao['CD_TIPO_MODALIDADE']
            
            folder_name = f"{org}_{num}_{ano}_{mod}"
            os.makedirs(f'licitacoes/{folder_name}', exist_ok=True)
            os.makedirs(f'licitacoes/{folder_name}/lotes', exist_ok=True)

            with open(f'licitacoes/{folder_name}/link.txt', 'w') as f:
                f.write(licitacao['LINK_LICITACON_CIDADAO'])

            lotes = lote_df[
                (lote_df['CD_ORGAO'] == org) &
                (lote_df['NR_LICITACAO'] == num) &
                (lote_df['ANO_LICITACAO'] == ano) &
                (lote_df['CD_TIPO_MODALIDADE'] == mod)
            ]

            for _, lote in lotes.iterrows():
                itens = item_df[
                    (item_df['CD_ORGAO'] == lote['CD_ORGAO']) &
                    (item_df['NR_LICITACAO'] == lote['NR_LICITACAO']) &
                    (item_df['ANO_LICITACAO'] == lote['ANO_LICITACAO']) &
                    (item_df['CD_TIPO_MODALIDADE'] == lote['CD_TIPO_MODALIDADE']) &
                    (item_df['NR_LOTE'] == lote['NR_LOTE'])
                ]
                
                itens.to_csv(f'licitacoes/{folder_name}/lotes/{lote["NR_LOTE"]}.csv', index=False)

if __name__ == "__main__":
    Desafio02().iniciar()