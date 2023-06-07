# Análise dos Bairros de São Paulo

Para ver os Notebooks completos sugiro usar o site https://nbviewer.org/. Exemplo: https://nbviewer.org/github/galvsoliveira/ds_geoprofit_analysis/blob/main/Notebooks/4__exploratory_data_analysis_sp.ipynb. Onde podemos ver os gráficos do Folium.

Para rodar os arquivos do repositório primeiro rode os seguintes comandos em ordem (ubuntu 22.04, python 3.10):
```
- sudo apt-get install pkg-config
- sudo apt-get install libcairo2-dev
- pip install -r requirements.txt
```

Este projeto tem como objetivo realizar uma análise dos bairros da cidade de São Paulo para estimar o faturamento, classificar o potencial e segmentar os bairros de acordo com a renda e idade, a fim de identificar aqueles com maior aderência ao público-alvo.

## Dados

Foram fornecidos dados de faturamento e potencial dos bairros do Rio de Janeiro, juntamente com os dados de sociodemografia dos bairros do Rio de Janeiro e São Paulo. Esses dados estão disponíveis em um arquivo raw_data.xlsx na pasta Data.

## Desafios

O desafio proposto envolve as seguintes tarefas:

1. Estimar o faturamento que uma loja teria em cada um dos bairros de São Paulo.
2. Classificar o potencial de cada bairro como Alto, Médio ou Baixo.
3. Segmentar os bairros de São Paulo de acordo com a renda e idade, e indicar aqueles com maior aderência ao público-alvo.

## Solução

Foram analisados os dados fornecidos e aplicado técnicas de ciência de dados para atender aos desafios apresentados.

### 1. Estimativa de Faturamento

Foi realizada uma análise dos dados de faturamento e sociodemografia dos bairros de São Paulo, utilizando técnicas de machine learning (regressão linear) para estimar o faturamento que uma loja teria em cada um dos bairros utilizando como base os dados sociodemograficos e de faturamento nos bairro do Rio de Janeiro. Foi obtido 93% de acurácia nos dados de teste.

![image](https://github.com/galvsoliveira/ds_geoprofit_analysis/assets/95829723/bbfd875d-df58-4a4e-8e58-4d428c6731de)

### 2. Classificação de Potencial

Utilizando como base o potencial dos bairros do Rio de Janeiro, foi feita uma classificação do potencial de São Paulo usando técnicas de classificação de Machine Learning para comparar as duas cidades e ver quão vantajoso será São Paulo em relação ao Rio de Janeiro. Após isso, foi feito uma análise comparativa com os dados dos bairros de São Paulo para classificar o potencial de cada bairro como Alto, Médio ou Baixo. Foram utilizadas técnicas de agrupamento e análise exploratória para identificar padrões e características dos bairros.

![image](https://github.com/galvsoliveira/ds_geoprofit_analysis/assets/95829723/7c0ee7a1-08a4-428d-b357-50340ca5b262)

### 3. Segmentação por Renda e Idade

Foi realizada uma segmentação dos bairros de São Paulo com base nos dados de renda e idade, a fim de identificar aqueles com maior aderência ao público-alvo da empresa alimentícia. Foram utilizadas técnicas de agrupamento e visualização de dados para identificar grupos de bairros com características semelhantes, onde foi possível construir uma tabela dinâmica e um gráfico dos bairros com melhor aderência.

![image](https://github.com/galvsoliveira/ds_geoprofit_analysis/assets/95829723/9d67e875-a8f1-470f-b5f4-f395e1e077ed)


## Resultados

Como resultado da análise, foi gerado um documento contendo as conclusões e insights obtidos durante a análise dos bairros de São Paulo. O documento foi elaborado de forma a ser compreensível para pessoas não técnicas, utilizando gráficos, tabelas e descrições para apresentar as conclusões de forma clara e objetiva. Destacamos que o bairro com maior aderência do público alvo foi o bairro de Moema.
![image](https://github.com/galvsoliveira/ds_geoprofit_analysis/assets/95829723/1a2dd245-8c61-4f04-a12f-1d435e65f4cc)

## Melhorias

Precisamos destacar que a forma de determinação do potencial para os bairros de São Paulo não foi a melhor possível e que seria interessante utilizarmos técnicas de agrupamento, como K-Means para determinar com maior eficácia os potenciais. Além disso, podemos enriquecer nossa análise com outros dados podem ser retirados de fontes públicas e privadas para melhorar nosso estudo, como:
- Dados de análise de concorrência.
- Dados de transporte público na localidade.
- Dados sobre informações geográficas da região, para podermos calcular a densidade populacional.
