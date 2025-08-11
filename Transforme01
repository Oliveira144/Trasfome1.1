import streamlit as st
import json
import os
import time
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import deque

# --- CLASSE PARA GERENCIAR O ESTADO E A LÓGICA ---
class PredictiveAnalyzer:
    def __init__(self):
        # Mapeamento de cores para emojis e nomes
        self.emoji_map = {'C': '🔴', 'V': '🔵', 'E': '🟡'}
        self.color_names = {'C': 'Vermelho', 'V': 'Azul', 'E': 'Empate'}
        self.color_to_num = {'C': 0, 'V': 1, 'E': 2}
        self.num_to_color = {0: 'C', 1: 'V', 2: 'E'}
        
        # Parâmetros do modelo
        self.feature_window_size = 10  # Janela de resultados para a previsão
        self.training_threshold = 20   # Mínimo de resultados para treinar o modelo
        
        # Estado inicial (sobrescrito pelo arquivo de dados se existir)
        self.history = []
        self.signals = []
        self.performance = {'total': 0, 'hits': 0, 'misses': 0}
        self.analysis = {
            'patterns': [],
            'riskLevel': 'Baixo',
            'manipulation': 'Nenhuma',
            'prediction': None,
            'confidence': 0,
            'recommendation': 'Observar'
        }
        self.model = None
        self.model_accuracy = 0.0
        
        # Carrega dados persistentes na inicialização
        self.load_data()

    # --- MÉTODOS DE GERENCIAMENTO DE DADOS PERSISTENTES ---
    def load_data(self):
        """Carrega dados do arquivo JSON."""
        if os.path.exists('analyzer_data.json'):
            with open('analyzer_data.json', 'r') as f:
                try:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.signals = data.get('signals', [])
                    self.performance = data.get('performance', {'total': 0, 'hits': 0, 'misses': 0})
                except json.JSONDecodeError:
                    st.warning("Arquivo de dados corrompido. Reiniciando o histórico.")
                    self.history = []
                    self.signals = []
                    self.performance = {'total': 0, 'hits': 0, 'misses': 0}

    def save_data(self):
        """Salva o estado atual no arquivo JSON."""
        data = {
            'history': self.history,
            'signals': self.signals,
            'performance': self.performance
        }
        with open('analyzer_data.json', 'w') as f:
            json.dump(data, f, indent=4)

    # --- MÉTODOS DE AÇÃO DO USUÁRIO ---
    def add_outcome(self, outcome):
        """Adiciona um novo resultado e executa a análise."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history.append({'result': outcome, 'timestamp': timestamp})
        
        self.verify_previous_prediction(outcome)
        self.analyze_data()
        
        if self.analysis['prediction'] is not None:
            self.signals.append({
                'time': timestamp,
                'patterns': self.analysis['patterns'],
                'prediction': self.analysis['prediction'],
                'correct': None
            })

        self.save_data()

    def undo_last(self):
        """Desfaz o último resultado adicionado."""
        if self.history:
            self.history.pop()
            
            if self.signals and self.signals[-1].get('correct') is None:
                self.signals.pop()

            if self.history:
                self.analyze_data()
            else:
                self.analysis = {
                    'patterns': [], 'riskLevel': 'Baixo', 'manipulation': 'Nenhuma',
                    'prediction': None, 'confidence': 0, 'recommendation': 'Observar'
                }
                self.model = None
                self.model_accuracy = 0.0
                
            self.save_data()
            return True
        return False
        
    def clear_history(self, hard_reset=False):
        """Limpa todo o histórico, sinais e métricas de desempenho."""
        self.history = []
        self.signals = []
        self.performance = {'total': 0, 'hits': 0, 'misses': 0}
        self.analysis = {
            'patterns': [], 'riskLevel': 'Baixo', 'manipulation': 'Nenhuma',
            'prediction': None, 'confidence': 0, 'recommendation': 'Observar'
        }
        self.model = None
        self.model_accuracy = 0.0
        self.save_data()
        
    # --- MÉTODOS DE ANÁLISE COM MACHINE LEARNING ---
    def _prepare_data_for_ml(self):
        """Prepara os dados do histórico para o treinamento do modelo."""
        numeric_results = [self.color_to_num[d['result']] for d in self.history]
        X, y = [], []
        
        # Cria janelas de features e labels
        for i in range(len(numeric_results) - self.feature_window_size):
            features = numeric_results[i : i + self.feature_window_size]
            label = numeric_results[i + self.feature_window_size]
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)

    def _train_model(self):
        """Treina o modelo de machine learning com o histórico de dados."""
        X, y = self._prepare_data_for_ml()
        
        if len(X) < 2:
            self.model = None
            self.model_accuracy = 0.0
            return

        # Treina um modelo Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Divide os dados para obter uma estimativa de acurácia
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.model_accuracy = self.model.score(X_test, y_test)
        
    def _make_ml_prediction(self):
        """Usa o modelo treinado para fazer uma previsão."""
        if self.model is None:
            return None, 0

        # Pega a última janela de resultados
        last_results_num = [self.color_to_num[d['result']] for d in self.history[-self.feature_window_size:]]
        
        # Usa o modelo para prever
        prediction_num = self.model.predict([last_results_num])[0]
        prediction_color = self.num_to_color[prediction_num]
        
        # Obtém a confiança da previsão
        probabilities = self.model.predict_proba([last_results_num])[0]
        confidence = probabilities[prediction_num] * 100
        
        return prediction_color, confidence

    # --- MÉTODO PRINCIPAL DE ANÁLISE ---
    def analyze_data(self):
        """Executa a análise completa, incluindo o treinamento do modelo."""
        data = self.history
        
        # Lógica de análise de risco e manipulação permanece a mesma
        patterns = self.detect_patterns(data)
        risk_level = self.assess_risk(data)
        manipulation = self.detect_manipulation(data)
        
        # --- NOVO: LÓGICA DE PREVISÃO COM ML ---
        if len(data) >= self.training_threshold:
            if self.model is None or len(data) % 5 == 0:  # Retreina o modelo a cada 5 rodadas para se adaptar
                self._train_model()
            
            prediction_color, confidence = self._make_ml_prediction()
        else:
            prediction_color, confidence = None, 0
            
        self.analysis = {
            'patterns': patterns,
            'riskLevel': risk_level,
            'manipulation': manipulation,
            'prediction': prediction_color,
            'confidence': confidence,
            'recommendation': self.get_recommendation(risk_level, manipulation, confidence)
        }

    def detect_patterns(self, data):
        """Detecta padrões no histórico (lógica do código anterior mantida como um recurso informativo)."""
        # A lógica de detecção de padrões é mantida para ser exibida na interface,
        # mas não influencia diretamente a previsão do modelo de ML.
        patterns = []
        if len(data) < 2: return patterns
        
        results = [d['result'] for d in data]

        current_streak = 1
        current_color = results[-1]
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current_color:
                current_streak += 1
            else:
                break
        if current_streak >= 2:
            patterns.append({
                'type': 'streak',
                'color': current_color,
                'length': current_streak,
                'description': f"{current_streak}x {self.color_names[current_color]} seguidos"
            })
        
        if len(results) >= 4:
            last4 = results[-4:]
            if last4[0] == last4[1] and last4[2] == last4[3] and last4[0] != last4[2]:
                patterns.append({'type': '2x2', 'description': 'Padrão 2x2 (Ex: CCVV)'})
                
        if len(results) >= 3 and results[-1] == results[-2] and results[-2] == results[-3]:
            patterns.append({'type': 'triple_rep', 'description': 'Padrão de repetição (Ex: CVV)'})
            
        return patterns

    def assess_risk(self, data):
        """Avalia o nível de risco com base no histórico."""
        if len(data) < 1: return 'Baixo'
        results = [d['result'] for d in data]
        risk_score = 0
        
        max_streak = 1
        if results:
            current_streak = 1
            current_color = results[0]
            for i in range(1, len(results)):
                if results[i] == current_color:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
                    current_color = results[i]
        
        if max_streak >= 5: risk_score += 40
        elif max_streak >= 4: risk_score += 25
        elif max_streak >= 3: risk_score += 10
        
        tie_streak = 0
        for r in reversed(results):
            if r == 'E':
                tie_streak += 1
            else:
                break
        if tie_streak >= 2: risk_score += 30

        if risk_score >= 50: return 'Alto'
        if risk_score >= 25: return 'Médio'
        return 'Baixo'

    def detect_manipulation(self, data):
        """Detecta possíveis sinais de manipulação."""
        if len(data) < 1: return 'Nenhuma'
        results = [d['result'] for d in data]
        manipulation_score = 0
        
        if len(results) > 0 and results.count('E') / len(results) > 0.25:
            manipulation_score += 30
        
        if len(results) >= 6:
            recent6 = results[-6:]
            p1, p2 = recent6[:3], recent6[3:]
            if len(set(p1)) == 1 and len(set(p2)) == 1 and p1[0] != p2[0]:
                manipulation_score += 25

        if manipulation_score >= 40: return 'Alta'
        if manipulation_score >= 20: return 'Média'
        return 'Nenhuma'
        
    def get_recommendation(self, risk, manipulation, confidence):
        """Gera uma recomendação com base nos dados de análise."""
        if risk == 'Alto' or manipulation in ['Média', 'Alta']:
            return 'Evitar'
        if confidence > 75:
            return 'Apostar'
        return 'Observar'

    def verify_previous_prediction(self, current_outcome):
        """Verifica se a previsão anterior foi correta e atualiza as métricas."""
        for i in reversed(range(len(self.signals))):
            signal = self.signals[i]
            if signal.get('correct') is None:
                self.performance['total'] += 1
                if signal['prediction'] == current_outcome:
                    self.performance['hits'] += 1
                    signal['correct'] = "✅"
                else:
                    self.performance['misses'] += 1
                    signal['correct'] = "❌"
                return

    def get_accuracy(self):
        """Calcula a acurácia das previsões."""
        if self.performance['total'] == 0:
            return 0.0
        return (self.performance['hits'] / self.performance['total']) * 100

# --- INTERFACE DO USUÁRIO STREAMLIT ---
st.set_page_config(page_title="Sistema de Análise Preditiva - Versão Inteligente", layout="wide", page_icon="🎰")
st.title("🎰 Sistema de Análise Preditiva - Cassino")
st.markdown("---")

# Inicialização do analisador na sessão
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = PredictiveAnalyzer()

analyzer = st.session_state.analyzer

# --- SEÇÃO DE ENTRADA DE DADOS E CONTROLES ---
st.subheader("Entrada de Resultados")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔴 Vermelho (C)", use_container_width=True, type="primary"):
        analyzer.add_outcome('C')
        st.rerun()
with col2:
    if st.button("🔵 Azul (V)", use_container_width=True, type="primary"):
        analyzer.add_outcome('V')
        st.rerun()
with col3:
    if st.button("🟡 Empate (E)", use_container_width=True, type="primary"):
        analyzer.add_outcome('E')
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
cols_controls = st.columns(2)
with cols_controls[0]:
    if st.button("↩️ Desfazer Último", use_container_width=True, type="secondary"):
        analyzer.undo_last()
        st.rerun()
with cols_controls[1]:
    if st.button("🗑️ Limpar Tudo", use_container_width=True, type="secondary"):
        analyzer.clear_history()
        st.rerun()

st.markdown("---")

# --- VISUALIZAÇÃO DE ANÁLISE E RECOMENDAÇÃO ---
st.subheader("📈 Análise Atual")
analysis = analyzer.analysis

if len(analyzer.history) < analyzer.training_threshold:
    st.info(f"O modelo de inteligência artificial está aprendendo... Adicione pelo menos **{analyzer.training_threshold}** resultados para receber as primeiras sugestões.")
else:
    if analysis['prediction']:
        display_prediction = analyzer.emoji_map.get(analysis['prediction']) + " " + analyzer.color_names.get(analysis['prediction'], "...")
        bg_color_prediction = ""
        if analysis['prediction'] == 'C': bg_color_prediction = "rgba(255, 0, 0, 0.2)"
        elif analysis['prediction'] == 'V': bg_color_prediction = "rgba(0, 0, 255, 0.2)"
        else: bg_color_prediction = "rgba(255, 255, 0, 0.2)"

        st.markdown(f"""
        <div style="
            background: {bg_color_prediction};
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            border: 2px solid #fff;
        ">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 10px;">
                Sugestão para a Próxima Rodada:
            </div>
            <div style="font-size: 40px; font-weight: bold; color: #fff; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                {display_prediction}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Confiança:** {analysis['confidence']:.2f}%")
        st.write(f"**Recomendação:** {analysis['recommendation']}")
        st.write(f"**Nível de Risco:** {analysis['riskLevel']}")
        st.write(f"**Possível Manipulação:** {analysis['manipulation']}")

        if analysis['patterns']:
            st.write("### Padrões Detectados:")
            for p in analysis['patterns']:
                st.write(f"- {p['description']}")
    else:
        st.info("Nenhum resultado registrado ainda. Adicione resultados para iniciar a análise.")
        
st.markdown("---")

# --- MÉTRICAS DE DESEMPENHO E HISTÓRICO ---
st.subheader("📊 Métricas de Desempenho")
accuracy = analyzer.get_accuracy()
col_met1, col_met2, col_met3 = st.columns(3)
col_met1.metric("Acurácia (do Modelo)", f"{analyzer.model_accuracy*100:.2f}%" if analyzer.model_accuracy else "N/A")
col_met2.metric("Total de Previsões", analyzer.performance['total'])
col_met3.metric("Acertos", analyzer.performance['hits'])

st.markdown("---")

st.subheader("🎲 Histórico de Resultados")
if analyzer.history:
    max_results = 90
    recent_history = analyzer.history[-max_results:][::-1]
    
    lines = []
    for i in range(0, len(recent_history), 9):
        row = recent_history[i:i+9]
        emojis = [analyzer.emoji_map[r['result']] for r in row]
        lines.append(" ".join(emojis))
    
    for line in lines:
        st.markdown(f"<div style='font-size: 24px;'>**{line}**</div>", unsafe_allow_html=True)
else:
    st.info("Nenhum resultado inserido ainda.")

st.markdown("---")

st.subheader("📑 Últimas Sugestões Geradas")
if analyzer.signals:
    for signal in analyzer.signals[-5:][::-1]:
        display = analyzer.emoji_map.get(signal['prediction']) + " " + analyzer.color_names.get(signal['prediction'], "...")
        status = signal.get('correct', '...')
        bg_color = "rgba(0, 255, 0, 0.1)" if status == "✅" else "rgba(255, 0, 0, 0.1)" if status == "❌" else "rgba(128, 128, 128, 0.1)"
        
        st.markdown(f"""
        <div style="
            background: {bg_color};
            border-radius: 10px;
            padding: 12px;
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div><strong>Sinal para aposta em:</strong></div>
            <div style="font-size: 24px; font-weight: bold;">{display}</div>
            <div style="color: {'green' if status == '✅' else 'red' if status == '❌' else 'gray'}; font-weight: bold; font-size: 24px;">
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Registre resultados para que as sugestões e seu desempenho apareçam aqui.")

