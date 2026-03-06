#!/usr/bin/env python3
"""
거시경제학 계산기 - 주요 거시경제 지표 계산 도구

주요 기능:
- GDP 계산 (지출접근법, 소득접근법)
- 인플레이션율 계산
- 실업률 계산
- 승수효과 계산
- 이자율 관련 계산
"""

import math
import json
from datetime import datetime

class MacroEconCalculator:
    def __init__(self):
        self.results_history = []
    
    def gdp_expenditure_approach(self, consumption, investment, government_spending, net_exports):
        """지출접근법으로 GDP 계산: GDP = C + I + G + (X - M)"""
        gdp = consumption + investment + government_spending + net_exports
        result = {
            'method': '지출접근법 GDP',
            'calculation': f'{consumption} + {investment} + {government_spending} + {net_exports}',
            'result': gdp,
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return gdp
    
    def inflation_rate(self, current_price, previous_price):
        """인플레이션율 계산: ((현재가격 - 이전가격) / 이전가격) × 100"""
        if previous_price == 0:
            raise ValueError("이전 가격이 0일 수 없습니다")
        
        inflation = ((current_price - previous_price) / previous_price) * 100
        result = {
            'method': '인플레이션율',
            'calculation': f'(({current_price} - {previous_price}) / {previous_price}) × 100',
            'result': round(inflation, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return round(inflation, 2)
    
    def unemployment_rate(self, unemployed, labor_force):
        """실업률 계산: (실업자 수 / 노동력) × 100"""
        if labor_force == 0:
            raise ValueError("노동력이 0일 수 없습니다")
        
        unemployment = (unemployed / labor_force) * 100
        result = {
            'method': '실업률',
            'calculation': f'({unemployed} / {labor_force}) × 100',
            'result': round(unemployment, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return round(unemployment, 2)
    
    def multiplier_effect(self, marginal_propensity_to_consume):
        """승수효과 계산: 1 / (1 - MPC)"""
        if marginal_propensity_to_consume >= 1:
            raise ValueError("한계소비성향은 1보다 작아야 합니다")
        
        multiplier = 1 / (1 - marginal_propensity_to_consume)
        result = {
            'method': '승수효과',
            'calculation': f'1 / (1 - {marginal_propensity_to_consume})',
            'result': round(multiplier, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return round(multiplier, 2)
    
    def real_gdp(self, nominal_gdp, gdp_deflator):
        """실질 GDP 계산: (명목 GDP / GDP 디플레이터) × 100"""
        if gdp_deflator == 0:
            raise ValueError("GDP 디플레이터가 0일 수 없습니다")
        
        real = (nominal_gdp / gdp_deflator) * 100
        result = {
            'method': '실질 GDP',
            'calculation': f'({nominal_gdp} / {gdp_deflator}) × 100',
            'result': round(real, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return round(real, 2)
    
    def compound_interest(self, principal, rate, time):
        """복리 계산: P(1 + r)^t"""
        amount = principal * (1 + rate) ** time
        result = {
            'method': '복리 계산',
            'calculation': f'{principal} × (1 + {rate})^{time}',
            'result': round(amount, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return round(amount, 2)
    
    def present_value(self, future_value, discount_rate, periods):
        """현재가치 계산: FV / (1 + r)^t"""
        pv = future_value / (1 + discount_rate) ** periods
        result = {
            'method': '현재가치',
            'calculation': f'{future_value} / (1 + {discount_rate})^{periods}',
            'result': round(pv, 2),
            'timestamp': datetime.now().isoformat()
        }
        self.results_history.append(result)
        return round(pv, 2)
    
    def print_menu(self):
        """메뉴 출력"""
        print("\n" + "="*50)
        print("📊 거시경제학 계산기")
        print("="*50)
        print("1. GDP 계산 (지출접근법)")
        print("2. 인플레이션율 계산")
        print("3. 실업률 계산")
        print("4. 승수효과 계산")
        print("5. 실질 GDP 계산")
        print("6. 복리 계산")
        print("7. 현재가치 계산")
        print("8. 계산 기록 보기")
        print("9. 공식 가이드")
        print("0. 종료")
        print("="*50)
    
    def formula_guide(self):
        """공식 가이드 출력"""
        print("\n📚 거시경제학 주요 공식 가이드")
        print("="*60)
        print("\n💰 GDP 관련")
        print("• 지출접근법: GDP = C + I + G + (X - M)")
        print("  - C: 소비, I: 투자, G: 정부지출, X-M: 순수출")
        print("• 실질 GDP = (명목 GDP / GDP 디플레이터) × 100")
        
        print("\n📈 물가와 인플레이션")
        print("• 인플레이션율 = ((현재가격 - 이전가격) / 이전가격) × 100")
        print("• CPI = (현재 장바구니 비용 / 기준연도 장바구니 비용) × 100")
        
        print("\n👥 고용과 실업")
        print("• 실업률 = (실업자 수 / 경제활동인구) × 100")
        print("• 경제활동참가율 = (경제활동인구 / 생산가능인구) × 100")
        
        print("\n🔄 승수효과")
        print("• 정부지출승수 = 1 / (1 - MPC)")
        print("• 조세승수 = -MPC / (1 - MPC)")
        print("  - MPC: 한계소비성향")
        
        print("\n💸 이자와 투자")
        print("• 복리: A = P(1 + r)^t")
        print("• 현재가치: PV = FV / (1 + r)^t")
        print("="*60)
    
    def show_history(self):
        """계산 기록 보기"""
        if not self.results_history:
            print("\n계산 기록이 없습니다.")
            return
        
        print("\n📋 계산 기록")
        print("="*60)
        for i, record in enumerate(self.results_history[-10:], 1):  # 최근 10개만 표시
            print(f"{i}. {record['method']}")
            print(f"   계산: {record['calculation']}")
            print(f"   결과: {record['result']}")
            print(f"   시간: {record['timestamp'][:19]}")
            print("-" * 40)
    
    def run(self):
        """메인 실행 함수"""
        print("🎓 거시경제학 계산기에 오신 것을 환영합니다!")
        
        while True:
            self.print_menu()
            try:
                choice = input("\n원하는 기능을 선택하세요: ").strip()
                
                if choice == '1':
                    print("\n💰 GDP 계산 (지출접근법)")
                    c = float(input("소비(C): "))
                    i = float(input("투자(I): "))
                    g = float(input("정부지출(G): "))
                    nx = float(input("순수출(X-M): "))
                    result = self.gdp_expenditure_approach(c, i, g, nx)
                    print(f"✅ GDP = {result:,.2f}")
                
                elif choice == '2':
                    print("\n📈 인플레이션율 계산")
                    current = float(input("현재 가격: "))
                    previous = float(input("이전 가격: "))
                    result = self.inflation_rate(current, previous)
                    print(f"✅ 인플레이션율 = {result}%")
                
                elif choice == '3':
                    print("\n👥 실업률 계산")
                    unemployed = float(input("실업자 수: "))
                    labor_force = float(input("경제활동인구: "))
                    result = self.unemployment_rate(unemployed, labor_force)
                    print(f"✅ 실업률 = {result}%")
                
                elif choice == '4':
                    print("\n🔄 승수효과 계산")
                    mpc = float(input("한계소비성향(MPC, 0~1 사이): "))
                    result = self.multiplier_effect(mpc)
                    print(f"✅ 승수 = {result}")
                
                elif choice == '5':
                    print("\n💲 실질 GDP 계산")
                    nominal = float(input("명목 GDP: "))
                    deflator = float(input("GDP 디플레이터: "))
                    result = self.real_gdp(nominal, deflator)
                    print(f"✅ 실질 GDP = {result:,.2f}")
                
                elif choice == '6':
                    print("\n💸 복리 계산")
                    principal = float(input("원금: "))
                    rate = float(input("이자율(소수점, 예: 0.05): "))
                    time = float(input("기간: "))
                    result = self.compound_interest(principal, rate, time)
                    print(f"✅ 최종 금액 = {result:,.2f}")
                
                elif choice == '7':
                    print("\n💰 현재가치 계산")
                    fv = float(input("미래가치: "))
                    rate = float(input("할인율(소수점, 예: 0.05): "))
                    periods = float(input("기간: "))
                    result = self.present_value(fv, rate, periods)
                    print(f"✅ 현재가치 = {result:,.2f}")
                
                elif choice == '8':
                    self.show_history()
                
                elif choice == '9':
                    self.formula_guide()
                
                elif choice == '0':
                    print("\n👋 계산기를 종료합니다. 공부 화이팅!")
                    break
                
                else:
                    print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                
            except ValueError as e:
                print(f"❌ 입력 오류: {e}")
            except Exception as e:
                print(f"❌ 계산 오류: {e}")

if __name__ == "__main__":
    calculator = MacroEconCalculator()
    calculator.run()