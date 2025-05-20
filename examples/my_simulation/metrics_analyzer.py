# metrics_analyzer.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any

class MetricsAnalyzer:
    """
    Analizza le metriche raccolte durante la simulazione e genera report.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Inizializza l'analizzatore.
        
        Args:
            output_dir: Directory per l'output dei report
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Impostazioni di stile per i grafici
        sns.set(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })
    
    def analyze_simulation(self, metrics: Dict[str, pd.DataFrame], config: Dict[str, Any] = None):
        """
        Analizza i risultati della simulazione e genera report.
        
        Args:
            metrics: Dict di DataFrame con le metriche della simulazione
            config: Configurazione della simulazione (opzionale)
        """
        # Estrai i DataFrame specifici
        inv_df = metrics.get('invocations_df')
        allocation_df = metrics.get('allocation_df')
        node_util_df = metrics.get('node_utilization_df')
        
        if inv_df is None:
            print("Nessun dato sulle invocazioni disponibile")
            return
        
        # Timestamp per i nomi dei file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva i DataFrame in CSV
        for name, df in metrics.items():
            if df is not None:
                df.to_csv(f"{self.output_dir}/{name}_{timestamp}.csv", index=False)
        
        # Genera report di riepilogo
        self._generate_summary_report(metrics, timestamp, config)
        
        # Genera grafici
        self._plot_response_time_distribution(inv_df, timestamp)
        self._plot_execution_time_distribution(inv_df, timestamp)
        self._plot_wait_time_distribution(inv_df, timestamp)
        self._plot_response_time_by_function(inv_df, timestamp)
        self._plot_invocations_per_node(inv_df, timestamp)
        
        if node_util_df is not None:
            self._plot_node_utilization(node_util_df, timestamp)
        
        # Salva informazioni sulla configurazione
        if config:
            with open(f"{self.output_dir}/config_{timestamp}.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Report e grafici salvati in: {self.output_dir}")
    
    def _generate_summary_report(self, metrics: Dict[str, pd.DataFrame], timestamp: str, config: Dict[str, Any] = None):
        """Genera un report riassuntivo delle metriche principali."""
        inv_df = metrics.get('invocations_df')
        
        if inv_df is None:
            return
        
        # Calcola metriche
        total_invocations = len(inv_df)
        
        # Metriche di tempo
        time_metrics = {}
        for col in ['t_wait', 't_exec', 'response_time']:
            if col in inv_df.columns:
                time_metrics[col] = {
                    'mean': inv_df[col].mean(),
                    'min': inv_df[col].min(),
                    'max': inv_df[col].max(),
                    'median': inv_df[col].median(),
                    'p95': inv_df[col].quantile(0.95)
                }
        
        # Invocazioni per funzione
        func_counts = inv_df['function'].value_counts().to_dict() if 'function' in inv_df.columns else {}
        
        # Invocazioni per nodo
        node_counts = inv_df['node'].value_counts().to_dict() if 'node' in inv_df.columns else {}
        
        # Unisci tutti i dati
        summary = {
            'total_invocations': total_invocations,
            'time_metrics': time_metrics,
            'function_counts': func_counts,
            'node_counts': node_counts,
            'config': config
        }
        
        # Salva il report
        with open(f"{self.output_dir}/summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Genera anche una versione testuale del report
        with open(f"{self.output_dir}/summary_{timestamp}.txt", 'w') as f:
            f.write("=== RIEPILOGO SIMULAZIONE ===\n\n")
            f.write(f"Totale invocazioni: {total_invocations}\n\n")
            
            f.write("--- METRICHE DI TEMPO (ms) ---\n")
            for metric_name, values in time_metrics.items():
                readable_name = {
                    't_wait': 'Tempo di attesa',
                    't_exec': 'Tempo di esecuzione',
                    'response_time': 'Tempo di risposta'
                }.get(metric_name, metric_name)
                
                f.write(f"{readable_name}:\n")
                f.write(f"  Media: {values['mean']:.2f}\n")
                f.write(f"  Min: {values['min']:.2f}\n")
                f.write(f"  Max: {values['max']:.2f}\n")
                f.write(f"  Mediana: {values['median']:.2f}\n")
                f.write(f"  95° percentile: {values['p95']:.2f}\n\n")
            
            f.write("--- INVOCAZIONI PER FUNZIONE ---\n")
            for func, count in func_counts.items():
                f.write(f"{func}: {count}\n")
            f.write("\n")
            
            f.write("--- INVOCAZIONI PER NODO ---\n")
            for node, count in node_counts.items():
                f.write(f"{node}: {count}\n")
            f.write("\n")
            
            if config:
                f.write("--- CONFIGURAZIONE ---\n")
                f.write(json.dumps(config, indent=2))
    
    def _plot_response_time_distribution(self, df: pd.DataFrame, timestamp: str):
        """Genera un grafico della distribuzione del tempo di risposta."""
        if 'response_time' not in df.columns:
            return
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df['response_time'], kde=True, bins=30, color='skyblue')
        plt.title('Distribuzione del Tempo di Risposta')
        plt.xlabel('Tempo di risposta (ms)')
        plt.ylabel('Frequenza')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/response_time_distribution_{timestamp}.png")
        plt.close()
    
    def _plot_execution_time_distribution(self, df: pd.DataFrame, timestamp: str):
        """Genera un grafico della distribuzione del tempo di esecuzione."""
        if 't_exec' not in df.columns:
            return
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df['t_exec'], kde=True, bins=30, color='salmon')
        plt.title('Distribuzione del Tempo di Esecuzione')
        plt.xlabel('Tempo di esecuzione (ms)')
        plt.ylabel('Frequenza')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/execution_time_distribution_{timestamp}.png")
        plt.close()
    
    def _plot_wait_time_distribution(self, df: pd.DataFrame, timestamp: str):
        """Genera un grafico della distribuzione del tempo di attesa."""
        if 't_wait' not in df.columns:
            return
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df['t_wait'], kde=True, bins=30, color='lightgreen')
        plt.title('Distribuzione del Tempo di Attesa')
        plt.xlabel('Tempo di attesa (ms)')
        plt.ylabel('Frequenza')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/wait_time_distribution_{timestamp}.png")
        plt.close()
    
    def _plot_response_time_by_function(self, df: pd.DataFrame, timestamp: str):
        """Genera un grafico del tempo di risposta per funzione."""
        if 'response_time' not in df.columns or 'function' not in df.columns:
            return
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='function', y='response_time', data=df, palette='Set2')
        plt.title('Tempo di Risposta per Funzione')
        plt.xlabel('Funzione')
        plt.ylabel('Tempo di risposta (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/response_time_by_function_{timestamp}.png")
        plt.close()
    
    def _plot_invocations_per_node(self, df: pd.DataFrame, timestamp: str):
        """Genera un grafico del numero di invocazioni per nodo."""
        if 'node' not in df.columns:
            return
        
        plt.figure(figsize=(14, 6))
        node_counts = df['node'].value_counts().sort_index()
        sns.barplot(x=node_counts.index, y=node_counts.values, palette='viridis')
        plt.title('Numero di Invocazioni per Nodo')
        plt.xlabel('Nodo')
        plt.ylabel('Numero di invocazioni')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/invocations_per_node_{timestamp}.png")
        plt.close()
    
    def _plot_node_utilization(self, df: pd.DataFrame, timestamp: str):
        """Genera un grafico dell'utilizzo dei nodi nel tempo."""
        if df.empty or 'time' not in df.columns:
            return
        
        plt.figure(figsize=(14, 8))
        
        # Raggruppa per nodo
        nodes = df['node'].unique()
        for node in nodes:
            node_df = df[df['node'] == node]
            
            # Utilizza la prima metrica disponibile tra cpu, memory, network
            for metric in ['cpu_usage', 'memory_usage', 'network_usage']:
                if metric in df.columns:
                    sns.lineplot(x='time', y=metric, data=node_df, label=node)
                    break
        
        plt.title('Utilizzo dei Nodi nel Tempo')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Utilizzo (%)')
        plt.legend(title='Nodo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/node_utilization_{timestamp}.png")
        plt.close()