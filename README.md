# eacpp

# ビルドまでの手順

## 1. githubからクローン
```console
$ git clone https://github.com/hiro99988/eacpp.git
$ cd eacpp
$ git submodule update --init --recursive
```

## 2. ビルド

### 2.1 デバッグビルド
```console
$ cmake -S . -B ./out/build/debug -DCMAKE_BUILD_TYPE=Debug
$ cmake --build ./out/build/debug
```

### 2.2 リリースビルド
```console
$ cmake -S . -B ./out/build/release -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./out/build/release
```

# ファイル説明

実行ファイルは.out拡張子で生成される．
MoeadEx.cpp -> MoeadEx.out

## examplesディレクトリ

1. MoeadEx.cpp<br>
MOEA/DでZDT1を解く．実行時間とIGD+値が表示されるだけ．
2. MpMoeadEx.cpp<br>
MP-MOEA/DでZDT1を解く，実行時間とIGD+値が表示されるだけ．

## benchmarksディレクトリ

1. ParallelBenchmark.cpp<br>
MP-MOEA/Dのベンチマークテストを行う．実験を行うときはこのプログラムを動かす．
data/inputs/ParallelParameter.jsonにパラメータ設定を記述する．

以下，このプログラムを実行したときに出力されるファイルの説明である．

### elapsedTimes.csv

各試行の初期化にかかった時間と，初期化を除いた実行時間を示す．中身は以下の通りである．

```csv
trial,initialization_time_s,execution_time_s
1,0.004964642,0.051319858
...
```

### idealPoint/trial_{試行数}.csv

各世代の理想点を示す．rankはプロセッサのランクを示し，全てのプロセッサの各世代の理想点を記録している．中身は以下の通りである．

```csv
rank,generation,objective1,objective2
0,0,0.0093567018207555992,2.6272669642903597
...
```

### igdPlus/trial_{試行数}.csv

各世代のIGD+を示す．execution_time_sは初期化時間を除いた実行時間を示す．中身は以下の通りである．

```
```csv
generation,execution_time_s,igd+
0,0,2.2464700202468508
...
```

### objective/trial_{試行数}.csv

最終的な目的関数値を示す．isIndicatorCalculatedUsingNds=trueの場合，最終的に得られた非支配解の目的関数値を示し，isIndicatorCalculatedUsingNds=falseの場合，最終世代の目的関数値を示す．rankはどのランクのプロセッサが保持していた解であったかを示す．
中身は以下の通りである．

```
```csv
rank,objective1,objective2
9,1.5360561384482468e-06,1.9671703825355542
...
```

以下は，このプログラムを実行したときに出力されるファイルの構造例である．

```
out/data/250101-010101
├── ZDT1-30 # 問題ディレクトリ
│   ├── MP-MOEAD-Async # アルゴリズムディレクトリ
│   │   ├── elapsedTimes.csv # 各試行の実行時間のデータ
│   │   ├── idealPoint # 各試行における各世代の理想点のデータ
│   │   │   ├── trial_1.csv
│   │   │   └── ...
│   │   ├── igdPlus  # 各試行における各世代のIGD+のデータ
│   │   │   ├── trial_1.csv
│   │   │   └── ...
│   │   └── objective # 各試行における最終の目的関数値のデータ
│   │       ├── trial_1.csv
│   │       └── ...
│   └── MP-MOEAD-Sync # アルゴリズムディレクトリ
│       └── ...
├── ... # その他の問題ディレクトリ
└── parameter.json # 事前に設定したパラメータファイルのコピー
```

## data/inputs/ParallelParameter.json

- isIndicatorCalculatedUsingNds: bool<br>
非支配解を使用して指標（IGD+）を計算するかどうか<br>
true: 非支配解を使用する<br>
false: 非支配解ではなく，各世代におけてMP-MOEA/Dが保持していた解集団を使用する
- trial: int<br>
試行回数
- neighborhoodSize: int<br>
近傍サイズ
- crossoverRate: double<br>
交叉率
- divisionsNumOfWeightVectors: array<br>
MOEA/Dにおける重みベクトルの数（個体数）を計算するときに使用される大文字の$H$を，目的数ごとに設定する．
```json
"divisionsNumOfWeightVectors": [
    {
        "obj": 2,
        "value": 199
    },
    {
        "obj": 3,
        "value": 23
    }
]
```
- generationsNums: array<br>
世代数をlevelごとに設定する．problemsでlevelを設定することで，問題の種類・難易度ごとに世代数を変えることができる．
```json
"generationsNums": [
    {
        "level": 0,
        "value": 500
    },
    {
        "level": 1,
        "value": 1000
    }
]
```
- algorithms: array<br>
アルゴリズム設定．現在使用できるアルゴリズムはMP-MOEA/Dのみ，nameに指定するときはスラッシュ"/"を除外すること．nameの先頭の文字列からアルゴリズムを判別するため，MP-MOEADは必ず先頭に書く，その後ろは何でもよい．
```json
"algorithms": [
    {
        "name": "MP-MOEAD-Async",
        "isAsync": true,
        "migrationInterval": 1
    },
    {
        "name": "MP-MOEAD-Sync",
        "isAsync": false,
        "migrationInterval": 1
    }
],
```

- problems: array<br>
問題設定．現在使用できる問題はZDT1~4,6とDTLZ1~4のみ．ただし，DTLZは目的数3と5のみ実行可能，正解のパレートフロントを目的数3と5のみ用意しているためそれ以外の目的数だとIGD+の計算ができない．ZDTの場合は決定変数の数とレベル，DTLZの場合は決定変数・目的の数とレベルを設定する．nameは，ZDTの場合はZDT{問題番号}-{決定変数の数}，DTLZの場合はDTLZ{問題番号}-{目的数}-{決定変数の数}と設定すること．
```json
"problems": [
    {
        "name": "ZDT1-30",
        "decisionVariablesNum": 30,
        "level": 0
    },
    {
        "name": "ZDT1-100",
        "decisionVariablesNum": 30,
        "level": 1
    },
    {
        "name": "DTLZ1-3-7",
        "decisionVariablesNum": 7,
        "objectivesNum": 3,
        "level": 0
    }
]
```

## pythonディレクトリ

実験結果の分析やプロットにはpythonを用いる．

以下は，それらに使用するpythonファイルの説明である．

### all.py

```console
$ python3 python/all.py <ディレクトリパス>
```

以下のpythonファイルを全て実行する．

### all_gen_igd.py

```console
$ python3 python/all_gen_igd.py <ディレクトリパス>
```

各問題において，アルゴリズムごとに全ての試行のIGD+値をプロットする．横軸は世代，縦軸はIGD+値．
<ディレクトリパス>/results/all_gen_igd/ に図が保存される．

### all_objective.py

```console
$ python3 python/all_objective.py <ディレクトリパス>
```

各問題において，アルゴリズムごとに全ての試行の目的関数値をプロットする．2,3次元のみ対応．軸はそれぞれ目的関数値．
<ディレクトリパス>/results/all_objective/ に図が保存される．

### base.py

共通変数・関数などを置いている．

### ideal_point_measure.py

```console
$ python3 python/ideal_point_measure.py <ディレクトリパス> [<世代間隔>]
```

各問題において，全てのアルゴリズムの全試行の理想点のAIPD（全てのノードが持つ理想点と全体情報がわかっているとき
の理想点との平均距離）の平均値をプロットする．横軸は世代，縦軸は平均AIPD．折れ線の周りに薄い色が付いているのは，標準偏差の幅を示す．

<ディレクトリパス>/results/avg_aipd に図が保存され，その中の<問題名>ディレクトリには各アルゴリズムの平均AIPDのcsvファイルが保存される．csvファイルのヘッダーがobjective1,objective2,objective3となっているがこれは間違いで，正しくは世代,平均AIPD値,標準偏差である．

世代間隔とは，AIPDの平均を取る世代間隔である．指定しない場合は1となる．1を指定すれば1世代ごとの平均値をプロットし，例えば10を指定すればまず1世代ごとの平均AIPDを計算し，さらにその10世代ごとの平均値をプロットする．図がギザギザしすぎて見づらい場合に設定するとよいかもしれない．


### median_objective.py

```console
$ python3 python/median_objective.py <ディレクトリパス>
```

各問題について，各アルゴリズムのIGD+が中央値である試行の目的関数値をプロットする．2次元問題のみに対応．
**statistics.pyを事前に実行する必要がある．**
<ディレクトリパス>/results/median_objective/ に図が保存される．

### statistics.py

```console
$ python3 python/statistics.py <ディレクトリパス>
```

統計的なデータを計算・保存する．

<ディレクトリパス>/results/igd+/ にIGD+のアルゴリズム別の統計的データを保存し，その中のrankディレクトリには各問題・アルゴリズムにおける昇順に並べた各試行のIGD+
を保存している．rankは全試行のうち何番目に小さいIGD+の値だったか,trialは試行番号を示す．

<ディレクトリパス>/results/time/ に初期化時間を除く実行時間のアルゴリズム別の統計的データを保存し，その中のexecution_timedディレクトリには各問題・アルゴリズムにおける昇順に並べた各試行の実行時間
を保存している．rankは全試行のうち何番目に小さい実行時間の値だったか,trialは試行番号を示す．

<ディレクトリパス>/results/wilcoxon/ にIGD+と実行時間に対して行ったウィルコクソンの符号順位検定の結果の表を保存する．画像の表は，行のアルゴリズムが    '+': 行のアルゴリズムが列のアルゴリズムより有意に良い場合には'+'，行のアルゴリズムが列のアルゴリズムより有意に悪い場合には'-'，有意差ない場合には'~'が示され，左のマークは実行時間の検定結果，右のマークはIGD+の検定結果となっている．

### time_igd.py

```console
$ python3 python/time_igd <ディレクトリパス>
```

各問題について，各アルゴリズムの全試行のIGD+値の平均値をプロットする．横軸は実行時間，縦軸は平均IGD+．折れ線の周りに薄い色が付いているのは，標準偏差の幅を示す．
**statistics.pyを事前に実行する必要がある．**
<ディレクトリパス>/results/time_igd/ に図が保存され，その中の<問題名>ディレクトリには各アルゴリズムの平均IGD+のcsvファイルが保存される．csvファイルのヘッダーがobjective1,objective2,objective3となっているがこれは間違いで，正しくは実行時間,平均IGD+値,標準偏差である．
