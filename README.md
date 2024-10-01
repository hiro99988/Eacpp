├── benchmarks　- ベンチマーク関連のcppファイル

├── data - 事前に用意したデータ
    ├── ground_truth
    │   └── pareto_fronts
    └─ inputs - 実行ファイルの入力用
        └── benchmarks
├── examples - 今はMOEA/DとMP-MOEA/Dを動かすcppファイル
├── include - ヘッダーファイル
│   ├── Algorithms - MOEA/Dなどのアルゴリズム
│   ├── Crossovers
│   ├── Decompositions - 集約関数
│   ├── Individual
│   ├── Mutations
│   ├── Problems
│   ├── Repairs
│   ├── Rng
│   ├── Samplings
│   ├── Selections
│   └── Utils - 便利関数
├── out - 出力
│   └── data - 実験結果
│       ├── moead
│       └── mp_moead
│           ├── ideal_point
│           ├── objective
│           └── plots
│               ├── ideal_point
│               └── objective
├── scripts
│   └── plot - プロット用pythonファイル
├── src - cppファイル
│   ├── Algorithms
│   ├── Crossovers
│   ├── Decompositions
│   ├── Mutations
│   ├── Problems
│   ├── Rng
│   ├── Samplings
│   ├── Selections
│   └── Utils
└── tests - テスト用cppファイル
    ├── Algorithms - 今はテストできない
    ├── Crossovers
    ├── Decompositions
    ├── Individual - 部分的にテストした
    ├── Mutations
    ├── Problems
    ├── Repairs
    ├── Rng
    ├── Samplings
    ├── Selections
    └── Utils
