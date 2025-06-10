O script deverá:

1. Baixar o arquivo `licitacoes-consolidado-2024.csv.zip` disponível
   em [TCERS](https://dados.tce.rs.gov.br/dataset/licitacoes-consolidado-2024)
2. Extrair, usando python, os arquivos presentes no `.zip`
3. Extrair em formato de [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), as licitações
   do arquivo `licitacao.csv`
4. Extrair em formato de [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), os lotes do
   arquivo `lote.csv`
5. Extrair em formato de [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), os itens do
   arquivo `item.csv`
6. Limpar os demais arquivos
7. Criar uma pasta `licitacoes`
8. Para cada licitação no `DataFrame` de licitações, gerar uma pasta com a seguinte estrutura de nome:
   `{CD_ORGAO}_{NR_LICITACAO}_{ANO_LICITACAO}_{CD_TIPO_MODALIDADE}`
9. Dentro de cada pasta relacionada à licitação, deverá existir uma pasta `lotes`
10. Para cada lote da licitação, deverá ser criado um arquivo `{NR_LOTE}.csv`, contendo o número do lote atual.
11. Cada linha do arquivo `{NR_LOTE}.csv` será um item daquele lote em específico.
12. Ainda na pasta da licitação, crie um arquivo `link.txt`, contendo o link presente no `DataFrame` de licitações

___
Dicas:

* Para baixar o arquivo, use alguma biblioteca para fazer requisições HTTP, é possível baixar o arquivo assim.
* Não use iterações (`for`, `while`, etc) para buscar os registros nas tabelas, o pandas possui métodos prontos e
  *muito* mais rápidos do que iterações linha a linha.

