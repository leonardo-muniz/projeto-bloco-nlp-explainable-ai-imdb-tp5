# **Teste de Performance 5: Análise de Sentimento e Explicabilidade (XAI) em Reviews de Filmes**

Este repositório contém o projeto final (TP5) do bloco de Machine Learning, uma investigação aprofundada em análise de sentimento que vai além da simples classificação, focando na interpretabilidade e explicabilidade de modelos de NLP.

### **Motivação**
Enquanto modelos de Machine Learning como redes neurais e ensembles podem atingir alta acurácia, eles frequentemente operam como "caixas-pretas" (black-box), tornando difícil entender suas decisões. Este projeto aborda esse desafio, aplicando técnicas de **Explainable AI (XAI)** para desmistificar um classificador de sentimento, tornando-o transparente e interpretável.

### **Base de Dados**
O projeto utiliza o **"IMDB Dataset of 50k Movie Reviews"**, um recurso clássico para tarefas de análise de sentimento.

**Conteúdo:** 50.000 reviews de filmes da plataforma IMDB.

**Target:** Uma classificação binária indicando se o sentimento do review é `positivo` ou `negativo`.

**Fonte:** Kaggle IMDB Dataset

**Nota Técnica de Carregamento:** Devido à natureza real dos dados, foi necessário tratar linhas malformadas durante a leitura do CSV, utilizando os parâmetros `engine='python'` e `on_bad_lines='skip'` no pandas.

### Metodologia e Pipeline do Projeto

O projeto foi estruturado em um pipeline completo, combinando NLP, modelagem supervisionada, visualização e XAI.

**1. Pré-processamento e Vetorização (TF-IDF):**

- Limpeza do texto dos reviews (remoção de tags HTML, pontuação, stopwords, etc.).

- Criação de uma representação numérica dos textos utilizando a matriz TF-IDF (Term Frequency-Inverse Document Frequency), que pondera a importância das palavras.

**2. Modelagem de Tópicos com Latent Dirichlet Allocation (LDA):**

- Aplicação do LDA sobre os dados para descobrir tópicos latentes de forma não supervisionada.

- O número ótimo de tópicos foi selecionado com base em **métricas de coerência (Coherence Score)** para garantir que os tópicos fossem semanticamente significativos.

**3. Classificação de Sentimento:**

- Foram treinados e comparados dois modelos robustos de classificação: **RandomForest** e **XGBoost**.

- Os modelos foram otimizados utilizando **GridSearchCV** e **Validação Cruzada** para encontrar os melhores hiperparâmetros.

- A performance foi avaliada com métricas como **Acurácia**, **Precisão**, **Recall**, **F1-Score** e **AUC-ROC**.

**4. Visualização de Clusters com t-SNE:**

- Aplicação da técnica de redução de dimensionalidade **t-SNE** sobre a matriz TF-IDF para visualizar os reviews como pontos em um gráfico 2D.

- Essa visualização permite inspecionar visualmente os agrupamentos de documentos, colorindo-os por sentimento (positivo/negativo).

**5. Explicabilidade do Modelo com LIME e SHAP (XAI):**

  - **LIME (Local Interpretable Model-agnostic Explanations):** Utilizado para explicar previsões individuais, mostrando quais palavras em um review específico mais contribuíram para a sua classificação.

  - **SHAP (SHapley Additive exPlanations):** Empregado para obter uma visão mais profunda da importância das features (palavras).

    - Análise Global: Identificação das palavras que, em média, mais impactam as previsões do modelo em todo o dataset.

    - Análise Local com `force_plot`: Visualização das forças que "empurram" a previsão de um único review para "positivo" ou "negativo", oferecendo uma explicação detalhada e intuitiva.

### Conclusões e Análise dos Resultados

- **Tópicos (LDA):** Foram identificados 5 tópicos principais. A análise das palavras-chave de cada um revelou temas distintos, como um tópico geral sobre a experiência de assistir a um filme (Tópico 1), um focado em narrativas e personagens (Tópico 2 e 3), e um possivelmente relacionado a filmes de crime ou suspense (Tópico 5, com palavras como 'police', 'killer', 'episode').

- **Performance do Classificador:** O modelo XGBoost apresentou o melhor desempenho geral, alcançando uma acurácia de 86% e um AUC de 0.937, superando o RandomForest. Isso demonstra uma alta capacidade preditiva para a tarefa de análise de sentimento.

- **Visualização (t-SNE):** O gráfico t-SNE revelou duas nuvens de pontos correspondentes aos sentimentos positivo e negativo, mostrando uma separação visual clara, embora com uma zona central de sobreposição, indicando reviews mais neutros ou ambíguos.

- **Explicabilidade (SHAP/LIME):** As análises com LIME e SHAP foram capazes de detalhar as previsões de forma impressionante. Por exemplo, em um review negativo do filme 'Phil the Alien', LIME destacou palavras como 'problem', 'low' e 'budget' como as principais contribuintes para a classificação negativa, mesmo com a presença de palavras positivas como 'better' e 'interesting'. Os `force_plots` do SHAP confirmaram visualmente essa lógica, aumentando a confiança e a transparência do modelo.
