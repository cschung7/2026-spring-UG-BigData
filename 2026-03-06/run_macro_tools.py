#!/usr/bin/env python3
"""
거시경제학 학습 도구 런처
모든 도구를 한 곳에서 실행할 수 있는 메인 메뉴
"""

import os
import sys
import subprocess

def show_main_menu():
    """메인 메뉴 표시"""
    print("\n" + "="*70)
    print("🎓 거시경제학 완전정복 도구모음")
    print("="*70)
    print("📚 학습을 위한 종합 도구입니다!")
    print()
    print("1️⃣  계산기 실행 - GDP, 인플레이션, 실업률 등 계산")
    print("2️⃣  시각화 도구 - 그래프와 차트로 경제지표 분석") 
    print("3️⃣  스터디 가이드 열기 - 핵심 개념과 공식 정리")
    print("4️⃣  도구 설치 확인 - 필요한 패키지 설치")
    print("5️⃣  샘플 실행 - 모든 도구 기능 미리보기")
    print("0️⃣  종료")
    print("="*70)

def check_dependencies():
    """필요한 패키지 확인 및 설치"""
    required_packages = ['matplotlib', 'numpy', 'pandas']
    missing_packages = []
    
    print("\n🔍 패키지 확인 중...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 설치됨")
        except ImportError:
            print(f"❌ {package} - 누락")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 누락된 패키지: {', '.join(missing_packages)}")
        install = input("설치하시겠습니까? (y/n): ").lower().strip()
        
        if install == 'y':
            for package in missing_packages:
                print(f"설치 중: {package}")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package])
            print("✅ 모든 패키지가 설치되었습니다!")
        else:
            print("⚠️  일부 기능이 제한될 수 있습니다.")
    else:
        print("\n✅ 모든 필요한 패키지가 설치되어 있습니다!")

def run_calculator():
    """계산기 실행"""
    try:
        print("\n🧮 거시경제학 계산기를 실행합니다...")
        from macro_econ_calculator import MacroEconCalculator
        calculator = MacroEconCalculator()
        calculator.run()
    except ImportError as e:
        print(f"❌ 계산기 모듈을 불러올 수 없습니다: {e}")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")

def run_visualizer():
    """시각화 도구 실행"""
    try:
        print("\n📊 시각화 도구를 실행합니다...")
        from econ_visualizer import EconVisualizer
        visualizer = EconVisualizer()
        visualizer.run()
    except ImportError as e:
        print(f"❌ 시각화 모듈을 불러올 수 없습니다: {e}")
        print("💡 matplotlib, numpy, pandas 패키지를 설치해주세요.")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")

def open_study_guide():
    """스터디 가이드 열기"""
    guide_file = "macro_study_guide.md"
    
    if os.path.exists(guide_file):
        print(f"\n📚 스터디 가이드를 엽니다: {guide_file}")
        
        # 운영체제별로 다른 명령어 사용
        try:
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', guide_file])
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', guide_file])
            elif sys.platform.startswith('win'):  # Windows
                subprocess.run(['start', guide_file], shell=True)
            else:
                print(f"📁 파일 위치: {os.path.abspath(guide_file)}")
                print("💡 직접 파일을 열어서 확인해주세요.")
        except:
            print(f"📁 파일 위치: {os.path.abspath(guide_file)}")
            print("💡 마크다운 뷰어나 텍스트 에디터로 열어주세요.")
    else:
        print("❌ 스터디 가이드 파일을 찾을 수 없습니다.")

def run_sample_demo():
    """샘플 데모 실행"""
    print("\n🎬 샘플 데모를 실행합니다...")
    print("\n1️⃣ 계산기 샘플:")
    print("   GDP = 1500조원 (C:900 + I:300 + G:250 + NX:50)")
    print("   인플레이션율 = 2.5%")
    print("   실업률 = 3.2%")
    
    try:
        from macro_econ_calculator import MacroEconCalculator
        calc = MacroEconCalculator()
        
        # 샘플 계산들
        gdp = calc.gdp_expenditure_approach(900, 300, 250, 50)
        inflation = calc.inflation_rate(102.5, 100)
        unemployment = calc.unemployment_rate(800, 25000)
        multiplier = calc.multiplier_effect(0.8)
        
        print(f"   ✅ GDP: {gdp:,.0f}조원")
        print(f"   ✅ 인플레이션율: {inflation}%")
        print(f"   ✅ 실업률: {unemployment}%")
        print(f"   ✅ 승수: {multiplier}")
        
    except ImportError:
        print("   ❌ 계산기 모듈 누락")
    
    print("\n2️⃣ 시각화 샘플:")
    try:
        from econ_visualizer import EconVisualizer
        viz = EconVisualizer()
        print("   📊 경제지표 그래프 생성 가능!")
        
        response = input("\n   시각화 샘플을 보시겠습니까? (y/n): ").lower().strip()
        if response == 'y':
            viz.run_sample_analysis()
        
    except ImportError:
        print("   ❌ 시각화 모듈 누락 (matplotlib, numpy, pandas 필요)")
    
    print("\n3️⃣ 스터디 가이드: macro_study_guide.md 파일 확인")

def main():
    """메인 실행 함수"""
    print("🎓 거시경제학 학습 도구모음에 오신 것을 환영합니다!")
    
    while True:
        show_main_menu()
        
        try:
            choice = input("\n원하는 기능을 선택하세요: ").strip()
            
            if choice == '1':
                run_calculator()
            
            elif choice == '2':
                run_visualizer()
            
            elif choice == '3':
                open_study_guide()
            
            elif choice == '4':
                check_dependencies()
            
            elif choice == '5':
                run_sample_demo()
            
            elif choice == '0':
                print("\n👋 학습 도구를 종료합니다.")
                print("📚 거시경제학 공부 화이팅!")
                break
            
            else:
                print("❌ 잘못된 선택입니다. 0-5 사이의 숫자를 입력해주세요.")
        
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()