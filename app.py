from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Configurações para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do diretório para arquivos estáticos
templates = Jinja2Templates(directory="templates")

# Criar diretório de templates se não existir
os.makedirs("templates", exist_ok=True)

# Função para salvar o HTML
def save_template():
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write("""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise de Clientes e Campanhas</title>
    <style>
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #166088;
            --accent-color: #4fc3a1;
            --background-color: #f8f9fa;
            --text-color: #333;
            --light-shadow: rgba(0, 0, 0, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 2px 10px var(--light-shadow);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        main {
            padding: 2rem 0;
        }
        
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px var(--light-shadow);
            padding: 1.5rem;
            margin-bottom: 2rem;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        h2 {
            color: var(--primary-color);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent-color);
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        
        button:hover {
            background-color: #3da889;
        }
        
        .results-section {
            margin-top: 3rem;
        }
        
        .chart-container {
            margin: 2rem 0;
            text-align: center;
        }
        
        .chart {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px var(--light-shadow);
        }
        
        .cluster-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .cluster-card {
            background-color: #f0f4f8;
            border-left: 4px solid var(--primary-color);
            padding: 1rem;
            border-radius: 4px;
        }
        
        .cluster-name {
            font-weight: 700;
            color: var(--secondary-color);
            margin-bottom: 0.5rem;
        }
        
        .metric {
            margin: 0.5rem 0;
            display: flex;
            justify-content: space-between;
        }
        
        .metric-name {
            font-weight: 600;
        }
        
        .recommendations {
            margin-top: 2rem;
        }
        
        .recommendation-item {
            margin: 0.8rem 0;
            padding-left: 1rem;
            border-left: 2px solid var(--accent-color);
        }
        
        .loader {
            display: none;
            text-align: center;
            margin: 2rem 0;
        }
        
        .loader-spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid var(--accent-color);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            color: #d9534f;
            padding: 1rem;
            background-color: #f9e4e4;
            border-radius: 4px;
            margin-top: 1rem;
            display: none;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
        }
        
        th, td {
            padding: 0.75rem;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        
        th {
            background-color: var(--primary-color);
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        tr:hover {
            background-color: #e9ecef;
        }
        
        /* Responsividade */
        @media (max-width: 768px) {
            .cluster-info {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Análise de Clientes e Campanhas</h1>
            <p class="subtitle">Análise de clusterização e desempenho de campanhas de marketing</p>
        </div>
    </header>
    
    <main class="container">
        <section class="card">
            <h2>Upload de Arquivos</h2>
            <form action="/upload/" enctype="multipart/form-data" method="post" id="uploadForm">
                <div class="form-group">
                    <label for="transacoes">Arquivo de Transações:</label>
                    <input type="file" id="transacoes" name="transacoes" accept=".csv" required>
                </div>
                <div class="form-group">
                    <label for="campanhas">Arquivo de Campanhas:</label>
                    <input type="file" id="campanhas" name="campanhas" accept=".csv" required>
                </div>
                <button type="submit">Analisar Dados</button>
            </form>
            <div class="loader" id="loader">
                <div class="loader-spinner"></div>
                <p>Processando dados, por favor aguarde...</p>
            </div>
            <div class="error-message" id="errorMessage"></div>
        </section>
        
        <section class="results-section" id="resultsSection" style="display: none;">
            <div class="card">
                <h2>Clusterização de Clientes</h2>
                <div class="chart-container">
                    <img id="clusterChart" class="chart" alt="Gráfico de clusters de clientes">
                </div>
                <h3>Diagnóstico por Cluster</h3>
                <div class="cluster-info" id="clusterInfo">
                    <!-- Cluster information will be inserted here -->
                </div>
            </div>
            
            <div class="card">
                <h2>Análise de Campanhas</h2>
                <div class="chart-container">
                    <img id="campaignChart1" class="chart" alt="Gráfico de gasto médio por campanha">
                </div>
                <div class="chart-container">
                    <img id="campaignChart2" class="chart" alt="Gráfico de ROI por campanha">
                </div>
            </div>
            
            <div class="card">
                <h2>Impacto das Campanhas</h2>
                <div class="chart-container">
                    <img id="impactChart" class="chart" alt="Gráfico de impacto das campanhas">
                </div>
            </div>
            
            <div class="card">
                <h2>Modelo de Previsão</h2>
                <div class="chart-container">
                    <img id="predictionChart" class="chart" alt="Gráfico de previsão vs real">
                </div>
            </div>
            
            <div class="card">
                <h2>Valor do Cliente ao Longo da Vida (CLV)</h2>
                <div class="chart-container">
                    <img id="clvChart" class="chart" alt="Gráfico de distribuição do CLV">
                </div>
            </div>
            
            <div class="card">
                <h2>Clientes de Alto Valor</h2>
                <div id="highValueCustomers">
                    <!-- High-value customer information will be inserted here -->
                </div>
                
                <h3 class="recommendations">Recomendações de Marketing</h3>
                <div id="recommendations">
                    <!-- Recommendations will be inserted here -->
                </div>
            </div>
        </section>
    </main>
    
    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const form = document.getElementById('uploadForm');
            const formData = new FormData(form);
            
            // Show loader and hide error message
            document.getElementById('loader').style.display = 'block';
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'none';
            
            fetch('/upload/', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Erro no processamento dos dados.');
                }
                return response.json();
            })
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                // Show results section
                document.getElementById('resultsSection').style.display = 'block';
                
                // Update charts
                document.getElementById('clusterChart').src = data.cluster_chart;
                document.getElementById('campaignChart1').src = data.campaign_chart1;
                document.getElementById('campaignChart2').src = data.campaign_chart2;
                document.getElementById('impactChart').src = data.impact_chart;
                document.getElementById('predictionChart').src = data.prediction_chart;
                document.getElementById('clvChart').src = data.clv_chart;
                
                // Update cluster information
                const clusterInfo = document.getElementById('clusterInfo');
                clusterInfo.innerHTML = '';
                
                data.cluster_diagnostics.forEach((cluster, index) => {
                    const clusterCard = document.createElement('div');
                    clusterCard.className = 'cluster-card';
                    
                    clusterCard.innerHTML = `
                        <div class="cluster-name">Cluster ${cluster.cluster}</div>
                        <div>Tipo de Cliente: ${cluster.tipo}</div>
                        <div class="metric">
                            <span class="metric-name">Frequência média:</span>
                            <span>${cluster.frequencia_compras}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Gasto total médio:</span>
                            <span>R$ ${cluster.total_gasto}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Dias desde última compra:</span>
                            <span>${cluster.ultima_compra}</span>
                        </div>
                    `;
                    
                    clusterInfo.appendChild(clusterCard);
                });
                
                // Update high-value customers
                const highValueCustomers = document.getElementById('highValueCustomers');
                if (data.high_value_customers.length > 0) {
                    let tableHTML = '<h3>Lista de Clientes de Alto Valor</h3>';
                    tableHTML += '<table>';
                    tableHTML += '<thead><tr><th>ID do Cliente</th><th>Total Gasto</th><th>Frequência de Compras</th></tr></thead>';
                    tableHTML += '<tbody>';
                    
                    data.high_value_customers.forEach(customer => {
                        tableHTML += `<tr>
                            <td>${customer.cliente_id}</td>
                            <td>R$ ${customer.total_gasto}</td>
                            <td>${customer.frequencia_compras}</td>
                        </tr>`;
                    });
                    
                    tableHTML += '</tbody></table>';
                    highValueCustomers.innerHTML = tableHTML;
                } else {
                    highValueCustomers.innerHTML = '<p>Nenhum cliente de alto valor identificado.</p>';
                }
                
                // Update recommendations
                const recommendations = document.getElementById('recommendations');
                recommendations.innerHTML = '';
                
                data.recommendations.forEach(recommendation => {
                    const recItem = document.createElement('div');
                    recItem.className = 'recommendation-item';
                    recItem.textContent = recommendation;
                    recommendations.appendChild(recItem);
                });
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loader').style.display = 'none';
                const errorMessage = document.getElementById('errorMessage');
                errorMessage.textContent = error.message || 'Erro ao processar os dados. Verifique se os arquivos estão no formato correto.';
                errorMessage.style.display = 'block';
            });
        });
    </script>
</body>
</html>
        """)

# Salvar o template HTML ao iniciar o aplicativo
save_template()

# Funções de análise
def plot_to_base64(plt):
    """Converte um gráfico matplotlib para base64 string para exibição HTML."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{image_base64}"

def analisar_dados(transacoes_df, campanhas_df):
    """Realiza a análise dos dados de transações e campanhas."""
    results = {}
    
    # Clusterização de Clientes
    clientes = transacoes_df.groupby('cliente_id').agg({
        'frequencia_compras': 'max',
        'total_gasto': 'max',
        'ultima_compra': 'max'
    }).reset_index()
    
    scaler = StandardScaler()
    clientes_scaled = scaler.fit_transform(clientes[['frequencia_compras', 'total_gasto', 'ultima_compra']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clientes['cluster'] = kmeans.fit_predict(clientes_scaled)
    
    pca = PCA(n_components=2)
    clientes[['pca1', 'pca2']] = pca.fit_transform(clientes_scaled)
    
    # Gráfico de clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clientes, x='pca1', y='pca2', hue='cluster', palette='Set2')
    plt.title('Clusters de Clientes')
    results['cluster_chart'] = plot_to_base64(plt)
    
    # Diagnóstico dos clusters
    cluster_diagnostico = clientes.groupby('cluster')[['frequencia_compras', 'total_gasto', 'ultima_compra']].mean().round(2).reset_index()
    
    # Classificação dos clusters
    cluster_diagnostics = []
    for _, row in cluster_diagnostico.iterrows():
        cluster_info = {
            'cluster': int(row['cluster']),
            'frequencia_compras': float(row['frequencia_compras']),
            'total_gasto': float(row['total_gasto']),
            'ultima_compra': float(row['ultima_compra'])
        }
        
        if row['frequencia_compras'] > 12 and row['total_gasto'] > 5000:
            cluster_info['tipo'] = "Cliente fiel e de alto valor"
        elif row['ultima_compra'] > 250:
            cluster_info['tipo'] = "Cliente inativo"
        else:
            cluster_info['tipo'] = "Cliente de valor médio e recorrência moderada"
        
        cluster_diagnostics.append(cluster_info)
    
    results['cluster_diagnostics'] = cluster_diagnostics
    
    # Merge do cluster com transações
    transacoes_df = pd.merge(transacoes_df, clientes[['cliente_id', 'cluster']], on='cliente_id', how='left')
    
    # Análise de preferência por campanhas
    preferencia_campanhas = transacoes_df.groupby('campanha').agg({
        'cliente_id': 'nunique',
        'valor_compra': 'sum',
        'frequencia_compras': 'sum',
        'total_gasto': 'sum'
    }).reset_index()
    
    # Junta com dados da campanha
    preferencia_campanhas = pd.merge(preferencia_campanhas, campanhas_df, left_on='campanha', right_on='nome_campanha', how='left')
    
    # Cálculo de métricas
    preferencia_campanhas['gasto_medio_por_cliente'] = preferencia_campanhas['total_gasto'] / preferencia_campanhas['cliente_id']
    preferencia_campanhas['roi_estimado'] = preferencia_campanhas['total_gasto'] / preferencia_campanhas['custo_campanha']
    
    # Gráficos de campanhas
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(data=preferencia_campanhas, x='campanha', y='gasto_medio_por_cliente', ax=ax)
    ax.set_title('Gasto Médio por Cliente por Campanha')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    results['campaign_chart1'] = plot_to_base64(plt)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.barplot(data=preferencia_campanhas, x='campanha', y='roi_estimado', ax=ax)
    ax.set_title('ROI Estimado por Campanha')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    results['campaign_chart2'] = plot_to_base64(plt)
    
    # Regressão Linear para impacto das campanhas
    transacoes_reg = transacoes_df.merge(campanhas_df, left_on='campanha', right_on='nome_campanha', how='left')
    features = ['custo_campanha', 'alcance', 'conversao']
    X = transacoes_reg[features]
    y = transacoes_reg['total_gasto']
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    coef_df = pd.DataFrame({'Variavel': features, 'Coeficiente': reg_model.coef_})
    
    # Gráfico de impacto das campanhas
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='Variavel', y='Coeficiente')
    plt.title('Impacto das Campanhas no Total Gasto')
    plt.ylabel('Coeficiente da Regressão')
    results['impact_chart'] = plot_to_base64(plt)
    
    # Modelo de previsão com características do cliente
    df = transacoes_df[['idade', 'renda_mensal', 'frequencia_compras', 'total_gasto']].copy()
    df = df.rename(columns={'renda_mensal': 'renda_anual'})
    X_reg = df[["idade", "renda_anual", "frequencia_compras"]]
    y_reg = df["total_gasto"]
    model = LinearRegression()
    model.fit(X_reg, y_reg)
    
    df["total_gasto_previsto"] = model.predict(X_reg)
    
    # Gráfico de previsão
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="total_gasto", y="total_gasto_previsto", data=df, color="blue")
    plt.title("Previsão de Total Gasto vs. Real")
    plt.xlabel("Total Gasto Real")
    plt.ylabel("Total Gasto Previsto")
    results['prediction_chart'] = plot_to_base64(plt)
    
    # Análise CLV
    clientes_clv = transacoes_df[['cliente_id', 'total_gasto']].drop_duplicates()
    clientes_clv.rename(columns={'total_gasto': 'clv'}, inplace=True)
    
    thresh_clv = clientes_clv['clv'].quantile(0.75)
    clientes_clv['segmento_valor'] = clientes_clv['clv'].apply(lambda x: 'Alto Valor' if x >= thresh_clv else 'Demais')
    
    # Gráfico CLV
    plt.figure(figsize=(12, 6))
    sns.histplot(data=clientes_clv, x='clv', hue='segmento_valor', bins=30, kde=True, palette='Set2')
    plt.title('Distribuição do CLV por Segmento (CLV = Total Gasto)')
    plt.xlabel('Customer Lifetime Value')
    plt.ylabel('Frequência')
    results['clv_chart'] = plot_to_base64(plt)
    
    # Clientes de alto gasto
    clientes_alto_gasto = transacoes_df[transacoes_df['total_gasto'] >= 60000][['cliente_id', 'total_gasto', 'frequencia_compras']].drop_duplicates().to_dict('records')
    results['high_value_customers'] = clientes_alto_gasto
    
    # Recomendações de marketing
    recommendations = [
        "Oferecer experiências exclusivas e personalizadas, como eventos VIP ou convites para lançamentos de produtos.",
        "Criar um programa de fidelidade premium com recompensas e benefícios exclusivos.",
        "Enviar comunicações personalizadas com ofertas especiais e produtos exclusivos.",
        "Desenvolver um sistema de recomendação de produtos baseado no histórico de compra de cada cliente.",
        "Oferecer atendimento personalizado e exclusivo, como um gerente de contas dedicado.",
        "Investir em campanhas de marketing direcionadas aos clientes de alto valor, com foco em produtos de luxo ou serviços exclusivos.",
        "Priorizar campanhas com ROI elevado, como aquelas que entregaram maior retorno por real investido.",
        "Reavaliar ou reformular campanhas com ROI baixo, focando em novos formatos ou incentivos como brindes, frete grátis, etc.",
        "Investir mais em campanhas com alto gasto médio por cliente, pois indicam maior valor percebido."
    ]
    results['recommendations'] = recommendations
    
    return results

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_files(transacoes: UploadFile = File(...), campanhas: UploadFile = File(...)):
    try:
        # Ler os arquivos CSV
        transacoes_content = await transacoes.read()
        campanhas_content = await campanhas.read()
        
        transacoes_df = pd.read_csv(io.BytesIO(transacoes_content))
        campanhas_df = pd.read_csv(io.BytesIO(campanhas_content))
        
        # Verificar se os dados têm as colunas necessárias
        required_transacoes_columns = ['cliente_id', 'frequencia_compras', 'total_gasto', 'ultima_compra', 'campanha', 'valor_compra', 'idade', 'renda_mensal']
        required_campanhas_columns = ['nome_campanha', 'custo_campanha', 'alcance', 'conversao']
        
        missing_transacoes = [col for col in required_transacoes_columns if col not in transacoes_df.columns]
        missing_campanhas = [col for col in required_campanhas_columns if col not in campanhas_df.columns]
        
        if missing_transacoes or missing_campanhas:
            missing_cols = {
                'transacoes': missing_transacoes,
                'campanhas': missing_campanhas
            }
            return {"error": "Faltam colunas necessárias nos arquivos", "missing_columns": missing_cols}
        
        # Realizar a análise
        results = analisar_dados(transacoes_df, campanhas_df)
        
        return results
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")
