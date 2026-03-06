#!/usr/bin/env python3
"""
거시경제학 시각화 도구 - 경제 지표 그래프 생성

주요 기능:
- 경제성장률 추이 그래프
- 인플레이션 추이 그래프
- 실업률과 인플레이션 관계 (필립스 곡선)
- GDP 구성 요소 파이차트
- 경제주기 시뮬레이션
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd
from datetime import datetime, timedelta

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class EconVisualizer:
    def __init__(self):
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def plot_gdp_growth(self, years, growth_rates, title="GDP Growth Rate"):
        """GDP 성장률 추이 그래프"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 선 그래프
        ax.plot(years, growth_rates, marker='o', linewidth=2.5, 
                color=self.colors[0], markersize=6, markerfacecolor='white',
                markeredgecolor=self.colors[0], markeredgewidth=2)
        
        # 0% 기준선 추가
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 그래프 꾸미기
        ax.set_title(f'📈 {title}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Growth Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 데이터 포인트에 값 표시
        for i, (year, rate) in enumerate(zip(years, growth_rates)):
            ax.annotate(f'{rate:.1f}%', (year, rate), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_inflation_unemployment(self, inflation_data, unemployment_data, title="Phillips Curve"):
        """필립스 곡선 (인플레이션 vs 실업률)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 산점도
        scatter = ax.scatter(unemployment_data, inflation_data, 
                           c=range(len(inflation_data)), cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='white', linewidth=1)
        
        # 추세선 추가
        z = np.polyfit(unemployment_data, inflation_data, 1)
        p = np.poly1d(z)
        ax.plot(unemployment_data, p(unemployment_data), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_title(f'📊 {title}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Unemployment Rate (%)', fontsize=12)
        ax.set_ylabel('Inflation Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 컬러바 추가
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Period', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def plot_gdp_components(self, consumption, investment, government, net_exports, title="GDP Components"):
        """GDP 구성요소 파이차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 파이차트
        components = [consumption, investment, government, net_exports]
        labels = ['Consumption (C)', 'Investment (I)', 'Government (G)', 'Net Exports (X-M)']
        
        wedges, texts, autotexts = ax1.pie(components, labels=labels, autopct='%1.1f%%',
                                          colors=self.colors[:4], startangle=90,
                                          explode=(0.05, 0.05, 0.05, 0.05))
        
        ax1.set_title(f'🥧 {title}', fontsize=14, fontweight='bold', pad=20)
        
        # 막대 차트
        ax2.bar(labels, components, color=self.colors[:4], alpha=0.8, edgecolor='white', linewidth=2)
        ax2.set_title('GDP Components (Bar Chart)', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Value', fontsize=12)
        
        # 값 표시
        for i, v in enumerate(components):
            ax2.text(i, v + max(components) * 0.01, f'{v:,.0f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_business_cycle(self, periods=20, title="Business Cycle Simulation"):
        """경기순환 시뮬레이션"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 시간 축 생성
        t = np.linspace(0, 4*np.pi, periods)
        
        # GDP (경기순환)
        gdp_trend = 1000 + 50*np.arange(periods)  # 장기 성장 추세
        gdp_cycle = gdp_trend + 100*np.sin(t)      # 순환적 변동
        
        ax1.plot(range(periods), gdp_trend, 'b--', label='Long-term Trend', linewidth=2)
        ax1.plot(range(periods), gdp_cycle, 'r-', label='Actual GDP', linewidth=2.5, marker='o')
        ax1.fill_between(range(periods), gdp_trend, gdp_cycle, alpha=0.3)
        ax1.set_title('📈 GDP and Business Cycle', fontsize=14, fontweight='bold')
        ax1.set_ylabel('GDP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 실업률 (GDP와 역상관)
        unemployment = 7 - 3*np.sin(t + np.pi)  # GDP와 반대 위상
        
        ax2.plot(range(periods), unemployment, 'g-', linewidth=2.5, marker='s')
        ax2.fill_between(range(periods), unemployment, alpha=0.3, color='green')
        ax2.set_title('👥 Unemployment Rate', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Unemployment (%)')
        ax2.grid(True, alpha=0.3)
        
        # 인플레이션
        inflation = 3 + 2*np.sin(t - np.pi/4)  # GDP보다 약간 지연
        
        ax3.plot(range(periods), inflation, 'm-', linewidth=2.5, marker='^')
        ax3.fill_between(range(periods), inflation, alpha=0.3, color='magenta')
        ax3.set_title('📊 Inflation Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Inflation (%)')
        ax3.grid(True, alpha=0.3)
        
        # 경기 국면 표시
        phases = ['Expansion', 'Peak', 'Recession', 'Trough']
        phase_colors = ['lightgreen', 'yellow', 'lightcoral', 'lightblue']
        
        for i in range(0, periods, periods//4):
            phase_idx = (i // (periods//4)) % 4
            ax1.axvspan(i, min(i + periods//4, periods-1), 
                       alpha=0.2, color=phase_colors[phase_idx], 
                       label=phases[phase_idx] if i == 0 else "")
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """대화형 대시보드 메뉴"""
        print("\n" + "="*60)
        print("📊 거시경제학 시각화 도구")
        print("="*60)
        print("1. GDP 성장률 추이 그래프")
        print("2. 필립스 곡선 (인플레이션 vs 실업률)")
        print("3. GDP 구성요소 분석")
        print("4. 경기순환 시뮬레이션")
        print("5. 샘플 데이터로 모든 그래프 보기")
        print("0. 종료")
        print("="*60)
    
    def run_sample_analysis(self):
        """샘플 데이터로 모든 분석 실행"""
        print("\n📊 샘플 데이터를 사용하여 모든 시각화를 보여드리겠습니다...\n")
        
        # 1. GDP 성장률
        years = list(range(2015, 2025))
        gdp_growth = [3.2, 2.8, 3.5, 2.1, -3.1, 5.7, 4.2, 3.8, 2.9, 3.1]
        self.plot_gdp_growth(years, gdp_growth, "한국 GDP 성장률 (2015-2024)")
        
        # 2. 필립스 곡선
        unemployment = [3.8, 4.2, 5.1, 3.9, 7.2, 4.1, 3.7, 3.2, 3.5, 3.8]
        inflation = [1.2, 0.8, 0.9, 2.1, 0.5, 2.8, 4.1, 3.2, 2.1, 1.8]
        self.plot_inflation_unemployment(inflation, unemployment, "한국 필립스 곡선")
        
        # 3. GDP 구성요소
        self.plot_gdp_components(1200, 400, 350, -50, "2024년 한국 GDP 구성 (조원)")
        
        # 4. 경기순환
        self.plot_business_cycle(24, "경기순환 시뮬레이션 (2년간)")
    
    def run(self):
        """메인 실행 함수"""
        print("📈 거시경제학 시각화 도구에 오신 것을 환영합니다!")
        
        while True:
            self.create_interactive_dashboard()
            
            try:
                choice = input("\n원하는 기능을 선택하세요: ").strip()
                
                if choice == '1':
                    print("\n📈 GDP 성장률 데이터를 입력하세요:")
                    years_input = input("연도들 (쉼표로 구분, 예: 2020,2021,2022): ").split(',')
                    years = [int(year.strip()) for year in years_input]
                    
                    rates_input = input("성장률들 (쉼표로 구분, 예: 3.2,2.1,4.5): ").split(',')
                    rates = [float(rate.strip()) for rate in rates_input]
                    
                    self.plot_gdp_growth(years, rates)
                
                elif choice == '2':
                    print("\n📊 필립스 곡선 데이터를 입력하세요:")
                    inflation_input = input("인플레이션율들 (쉼표로 구분): ").split(',')
                    inflation = [float(x.strip()) for x in inflation_input]
                    
                    unemployment_input = input("실업률들 (쉼표로 구분): ").split(',')
                    unemployment = [float(x.strip()) for x in unemployment_input]
                    
                    self.plot_inflation_unemployment(inflation, unemployment)
                
                elif choice == '3':
                    print("\n🥧 GDP 구성요소를 입력하세요:")
                    c = float(input("소비(C): "))
                    i = float(input("투자(I): "))
                    g = float(input("정부지출(G): "))
                    nx = float(input("순수출(X-M): "))
                    
                    self.plot_gdp_components(c, i, g, nx)
                
                elif choice == '4':
                    periods = int(input("\n경기순환 시뮬레이션 기간 (권장: 20-30): "))
                    self.plot_business_cycle(periods)
                
                elif choice == '5':
                    self.run_sample_analysis()
                
                elif choice == '0':
                    print("\n👋 시각화 도구를 종료합니다!")
                    break
                
                else:
                    print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                    
            except ValueError as e:
                print(f"❌ 입력 오류: 숫자를 올바르게 입력해주세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    visualizer = EconVisualizer()
    visualizer.run()