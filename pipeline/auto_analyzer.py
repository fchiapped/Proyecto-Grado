# 设置无界面后端，避免GUI锁死
import matplotlib
matplotlib.use("Agg")

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pipeline_manager import PipelineManager
from datetime import datetime
import gc

class AutoAnalyzer:
    """自动分析器 - 用户交互界面层"""
    
    def __init__(self, data_path: str = None):
        """初始化自动分析器"""
        self.project_root = Path(__file__).parent.parent
        self.pipeline = PipelineManager()
        self.results_dir = self.project_root / "resultados_analisis"
        self.results_dir.mkdir(exist_ok=True)
        
    def create_report_dir(self, plant_name: str) -> Path:
        """创建报告目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.results_dir / f"{plant_name}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        return report_dir

    def analyze_plant_data(self, data_file: str, do_outliers: bool = True,
                          do_lagunas: bool = True, do_ar: bool = False, 
                          do_diff: bool = False, do_drift: bool = False):
        """
        分析单个工厂的数据
        
        Args:
            data_file: 数据文件路径
            do_outliers: 是否执行异常值分析
            do_lagunas: 是否执行数据空档(Laguna)分析
            do_ar: 是否执行AR模型分析
            do_diff: 是否执行差分分析
            do_drift: 是否执行漂移分析
        """
        from time import time
        start_time = time()

        # 重新加载分析函数
        try:
            self.pipeline.reload_analysis_functions()
            print("Funciones de análisis recargadas (hot-reload) desde Analisis/funciones_analisis.py")
        except Exception as e:
            print(f"Advertencia: no se pudieron recargar las funciones de análisis. Error: {e}")
        
        # 提取工厂名称
        plant_name = Path(data_file).stem.split('_')[1]
        print(f"\n{'='*50}")
        print(f"Iniciando análisis de datos para {plant_name}...")
        
        # 创建报告目录
        report_dir = self.create_report_dir(plant_name)
        
        # 1. 数据加载
        df = self.pipeline.load_data(data_file)
        self._save_basic_info(df, report_dir)
        
        # 2. 数据质量分析（可选）
        if do_lagunas:
            self._analyze_data_quality(df, report_dir)
        
        # 3. 异常值分析（可选）
        if do_outliers:
            self._analyze_outliers(df, report_dir)
        
        # 4. AR 模型分析（可选）
        if do_ar:
            self._analyze_ar(df, report_dir)
        
        # 5. 差分分析（可选）
        if do_diff:
            self._analyze_diff(df, report_dir)
        
        # 6. Drift 分析（可选）
        if do_drift:
            self._analyze_drift(df, report_dir)
        
        end_time = time()
        print(f"\n¡Análisis finalizado! Tiempo: {end_time - start_time:.2f} s")
        print(f"Informe guardado en: {report_dir.absolute()}")
        
    def _save_basic_info(self, df: pd.DataFrame, report_dir: Path):
        """Guardar información básica de los datos"""
        info = self.pipeline.generate_basic_info(df)
        
        with open(report_dir / "1_informacion_basica.txt", "w", encoding='utf-8') as f:
            f.write("Información básica de los datos\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Forma de los datos: {info['shape']}\n\n")
            f.write("Tipos de datos:\n")
            for col, dtype in info['dtypes'].items():
                f.write(f"  {col}: {dtype}\n")
            f.write("\nValores faltantes:\n")
            for col, missing in info['missing_values'].items():
                if missing > 0:
                    f.write(f"  {col}: {missing}\n")
            
    def _analyze_data_quality(self, df: pd.DataFrame, report_dir: Path):
        """Analizar la calidad de los datos"""
        quality_report = self.pipeline.generate_data_quality_report(df)
        
        with open(report_dir / "2_reporte_calidad_datos.txt", "w", encoding='utf-8') as f:
            f.write("Reporte de calidad de datos\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Cobertura temporal:\n")
            f.write(f"Inicio: {quality_report['time_range']['start']}\n")
            f.write(f"Fin: {quality_report['time_range']['end']}\n\n")
            
            f.write("Integridad de datos:\n")
            comp = quality_report['completeness']
            f.write(f"Días con datos: {comp['dates_with_data']}\n")
            f.write(f"Días sin datos: {comp['dates_without_data']}\n")
            f.write(f"Porcentaje con datos: {comp['percentage_with_data']}%\n")
            f.write(f"Porcentaje sin datos: {comp['percentage_without_data']}%\n\n")
            
            # 显示缺失数据的日期范围（lagunas）
            lagunas = quality_report['lagunas']
            if lagunas:
                f.write("Periodos sin datos (Lagunas):\n")
                for inicio, fin in lagunas[:10]:
                    f.write(f"  De {inicio} a {fin}\n")
                if len(lagunas) > 10:
                    f.write(f"  ... {len(lagunas) - 10} periodos más\n")
                    
    def _analyze_outliers(self, df: pd.DataFrame, report_dir: Path):
        """分析异常值"""
        from time import time
        
        # 选择数值列
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_time']
        
        print(f"\nIniciando análisis de atípicos; total de variables: {len(numeric_cols)}")
        print("-" * 50)
        
        # 使用 pipeline 批量分析
        outlier_results = self.pipeline.analyze_outliers_batch(df, numeric_cols)
        
        # 生成图表
        for i, col in enumerate(numeric_cols, 1):
            start_time = time()
            print(f"[{i}/{len(numeric_cols)}] Analizando atípicos de {col}...", end=' ')
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(15, 5))
            plot_df = df.copy()
            if 'date_time' in plot_df.columns:
                plot_df['date_time'] = pd.to_datetime(plot_df['date_time'], errors='coerce')
            
            is_ph = 'pH' in col or 'ph' in col.lower()
            self.pipeline.analysis_funcs.plot_outliers(plot_df, col, ax=ax, ph=is_ph)
            
            # 保存图表
            fig.tight_layout()
            fig.savefig(report_dir / f"3_analisis_atipicos_{col}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            gc.collect()
            
            outlier_count = outlier_results['summary'][col]
            print(f"Listo! ({time() - start_time:.2f}s, {outlier_count} atípicos)")
        
        # 保存异常值CSV
        if outlier_results['all_outliers']:
            all_outliers_df = pd.concat(outlier_results['all_outliers'], ignore_index=True)
            all_outliers_df.to_csv(report_dir / "3_atipicos.csv", index=False, encoding='utf-8-sig')
            print(f"\nDatos de atípicos exportados a: 3_atipicos.csv (total {len(all_outliers_df)} filas)")
        else:
            print("\nNo se detectaron atípicos")
        
        # 保存统计
        with open(report_dir / "3_resumen_atipicos.txt", "w", encoding='utf-8') as f:
            f.write("Resumen de atípicos\n")
            f.write("=" * 50 + "\n\n")
            for col, count in outlier_results['summary'].items():
                f.write(f"{col}: {count} atípicos\n")
    
    def _analyze_ar(self, df: pd.DataFrame, report_dir: Path):
        """AR 模型分析（预留接口）"""
        print(f"\nIniciando análisis AR...")
        print("-" * 50)
        
        result = self.pipeline.run_ar_analysis()
        
        with open(report_dir / "4_reporte_AR.txt", "w", encoding='utf-8') as f:
            f.write("Reporte de modelo AR\n")
            f.write("=" * 50 + "\n\n")
            
            if result.get('status') == 'not_implemented':
                f.write("⚠️  La funcionalidad de análisis del modelo AR aún no está implementada\n")
                f.write("A la espera de completarlo en Modelos/AR_model.ipynb para integrarlo\n")
            else:
                f.write(str(result))
    
    def _analyze_diff(self, df: pd.DataFrame, report_dir: Path):
        """Análisis de diferencias centradas"""
        from time import time
        start_time = time()
        
        print(f"\nIniciando análisis de diferencias...")
        print("-" * 50)
        
        # Ejecutar análisis con parámetros por defecto
        result = self.pipeline.run_diff_analysis()
        
        # Guardar reporte
        with open(report_dir / "5_reporte_diferencias.txt", "w", encoding='utf-8') as f:
            f.write("Reporte de análisis de diferencias centradas\n")
            f.write("=" * 50 + "\n\n")
            
            if result.get('status') == 'success':
                f.write(f"✓ Análisis completado exitosamente\n")
                f.write(f"Columnas analizadas: {result['n_columnas_analizadas']}\n\n")
                
                # Resumen por columna
                for col, datos in result['resultados'].items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  • Outliers detectados: {datos['n_outliers']}\n")
                    f.write(f"  • Valores normales: {datos['n_normales']}\n")
                    f.write(f"  • Datos insuficientes: {datos['n_insuficientes']}\n")
                    f.write(f"  • Desviación estándar global: {datos['std_global']:.4f}\n")
                
                f.write(f"\nParámetros utilizados:\n")
                if result['resultados']:
                    primer_col = list(result['resultados'].keys())[0]
                    params = result['resultados'][primer_col]['parametros']
                    f.write(f"  • Umbral de nulos por día: {params['null_threshold']}\n")
                    f.write(f"  • Multiplicador de std: {params['std_multiplier']}\n")
                    f.write(f"  • Umbral de diferencia absoluta: {params['diff_absolute_threshold']}\n")
                
                # Guardar CSVs detallados
                for col, datos in result['resultados'].items():
                    csv_filename = report_dir / f"5_diff_{col}.csv"
                    datos['resultados'].to_csv(csv_filename, index=True, encoding='utf-8-sig')
                
                print(f"✓ Análisis completado en {time() - start_time:.2f}s")
                print(f"  → {result['n_columnas_analizadas']} columnas analizadas")
                
            else:
                f.write("⚠️  La funcionalidad de análisis de diferencias aún no está implementada\n")
                f.write("A la espera de completarlo en Modelos/DIFF_model.ipynb para integrarlo\n")
                print(f"⚠️ {result.get('message', 'Error desconocido')}")

    
    def _analyze_drift(self, df: pd.DataFrame, report_dir: Path):
        """Drift 分析（预留接口）"""
        print(f"\nIniciando análisis de deriva (drift)...")
        print("-" * 50)
        
        drift_result = self.pipeline.run_drift_analysis()
        
        with open(report_dir / "6_reporte_drift.txt", "w", encoding='utf-8') as f:
            f.write("Reporte de deriva de datos\n")
            f.write("=" * 50 + "\n\n")
            
            if drift_result.get('status') == 'not_implemented':
                f.write("⚠️  La funcionalidad de análisis de deriva (drift) aún no está implementada\n")
                f.write("A la espera de añadir la función correspondiente en funciones_analisis.py\n")
            else:
                f.write(f"Deriva detectada: {'Sí' if drift_result.get('drift_detected') else 'No'}\n")
                f.write(f"Puntaje de deriva: {drift_result.get('drift_score', 0):.4f}\n")
                if drift_result.get('drift_features'):
                    f.write("\nCaracterísticas con deriva:\n")
                    for feat in drift_result['drift_features']:
                        f.write(f"  - {feat}\n")


def main():
    """主交互界面"""
    print("="*70)
    print("Sistema automático de análisis de plantas de tratamiento de agua")
    print("="*70)
    
    analyzer = AutoAnalyzer()
    
    # 配置默认值
    config = {
        'outliers': True,
        'lagunas': True,
        'ar': False,
        'diff': False,
        'drift': False
    }
    
    while True:
        print("\n" + "="*70)
        print("🏠 Menú principal")
        print("="*70)
        print("  [1] Analizar archivo")
        print("  [2] Configuración de análisis")
        print("  [0] Salir")
        print("-"*70)
        
        main_choice = input("Seleccione una opción (número): ").strip()
        
        if main_choice == '0':
            print("\n👋 ¡Hasta luego!")
            break
        
        elif main_choice == '1':
            # ===== 文件选择和分析 =====
            data_dir = analyzer.project_root / "df_procesados"
            csv_files = list(data_dir.glob("df_planta_*.csv"))
            
            if not csv_files:
                print(f"❌ No se encontraron archivos de datos válidos en {data_dir}")
                input("\nPulse Enter para volver al menú principal...")
                continue
            
            print("\nArchivos disponibles:")
            for i, file in enumerate(csv_files, 1):
                print(f"  [{i}] {file.name}")
            print(f"  [0] Volver al menú principal")
            
            file_choice = input("\nSeleccione el archivo a analizar (número): ").strip()
            
            if file_choice == '0':
                continue
                
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(csv_files):
                    chosen_path = csv_files[file_idx]
                else:
                    print("❌ Número de archivo inválido")
                    input("\nPulse Enter para continuar...")
                    continue
            except ValueError:
                print("❌ Ingrese un número válido")
                input("\nPulse Enter para continuar...")
                continue
            
            print(f"\n{'='*70}")
            print(f"Iniciando análisis: {chosen_path.name}")
            print(f"Configuración actual: Atípicos={config['outliers']}, Lagunas={config['lagunas']}, "
                  f"AR={config['ar']}, DIFF={config['diff']}, Drift={config['drift']}")
            print(f"{'='*70}")

            try:
                analyzer.analyze_plant_data(
                    str(chosen_path),
                    do_outliers=config['outliers'],
                    do_lagunas=config['lagunas'],
                    do_ar=config['ar'],
                    do_diff=config['diff'],
                    do_drift=config['drift']
                )
            except Exception as e:
                print(f"Error analizando {chosen_path.name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
            finally:
                input("\nPulse Enter para volver al menú principal...")
        
        elif main_choice == '2':
            # ===== 进入分析设置子菜单 =====
            while True:
                print("\n" + "="*70)
                print("⚙️  Menú de configuración de análisis")
                print("="*70)
                print(f"  [1] Análisis de atípicos : {'✓ Activado' if config['outliers'] else '✗ Desactivado'} (con gráficos)")
                print(f"  [2] Análisis de lagunas  : {'✓ Activado' if config['lagunas'] else '✗ Desactivado'}")
                print(f"  [3] Análisis AR          : {'✓ Activado' if config['ar'] else '✗ Desactivado'} (reserva)")
                print(f"  [4] Análisis de diferencias : {'✓ Activado' if config['diff'] else '✗ Desactivado'} (reserva)")
                print(f"  [5] Análisis de deriva (drift) : {'✓ Activado' if config['drift'] else '✗ Desactivado'} (reserva)")
                print("  [0] Volver al menú principal")
                print("-"*70)
                
                setting_choice = input("Seleccione una opción para alternar (número): ").strip()
                
                if setting_choice == '0':
                    break
                elif setting_choice == '1':
                    config['outliers'] = not config['outliers']
                    print(f"✓ Análisis de atípicos {'Activado' if config['outliers'] else 'Desactivado'}")
                elif setting_choice == '2':
                    config['lagunas'] = not config['lagunas']
                    print(f"✓ Análisis de lagunas {'Activado' if config['lagunas'] else 'Desactivado'}")
                elif setting_choice == '3':
                    config['ar'] = not config['ar']
                    print(f"✓ Análisis AR {'Activado' if config['ar'] else 'Desactivado'}")
                elif setting_choice == '4':
                    config['diff'] = not config['diff']
                    print(f"✓ Análisis de diferencias {'Activado' if config['diff'] else 'Desactivado'}")
                elif setting_choice == '5':
                    config['drift'] = not config['drift']
                    print(f"✓ Análisis de deriva (drift) {'Activado' if config['drift'] else 'Desactivado'}")
                else:
                    print("❌ Opción no válida, inténtelo de nuevo.")
        
        else:
            print("❌ Opción no válida. Ingrese 0, 1 o 2.")

if __name__ == "__main__":
    main()
