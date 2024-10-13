- benchmarks　- ベンチマーク関連のcppファイル
- data - 事前に用意したデータ
  - ground_truth
    - pareto_fronts
  - inputs - 実行ファイルの入力用
    - benchmarks
- examples - 今はMOEA/DとMP-MOEA/Dを動かすcppファイル
- include - ヘッダーファイル
  - Algorithms - MOEA/Dなどのアルゴリズム
  - Crossovers
  - Decompositions - 集約関数
  - Individual
  - Mutations
  - Problems
  - Repairs
  - Rng - 乱数クラス
  - Samplings
  - Selections
  - Utils - 便利関数
- out - 出力
  - data - 実験結果
    - moead
    - mp_moead
      - ideal_point
      - objective
      - plots
        - ideal_point
        - objective
- scripts
  - plot - プロット用pythonファイル
- src - cppファイル
  - ...
- tests - テスト用cppファイル
  - Algorithms - 今はテストできない
  - ...
  - Individual - 部分的にテストした
  - ...