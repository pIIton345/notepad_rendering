# notepad_rendering
windowsのメモ帳、notepad.exeで画像・動画を表示できる。  
旧notepad.exeを使うことでカラー画像を表示できる。
テキストを10%表示にすると、文字が2ドットになってアンチエイリアスによって色がつく仕様を使った

# 使い方
1. テキストファイルを作成する。例としてout.txtとします。
2. notepad.exeで作成したテキストファイルを開きます。
3. コマンドプロンプトを開き、コマンドプロンプトでnp_render.py(np_render.exe)を実行する。
4. 表示したいファイルが画像ならnp_image.py(np_image.exe)を実行、動画ならnp_video.py(np_video.exe)を実行することでテキストファイルに書き込まれ、notepad.exeに表示されます。

pythonスクリプトを実行する場合、必要なパッケージをインストールしてください。
```
pip install -r requirements.txt
```
## np_render
```
np_render filepath
# out.txtを開くなら
np_render out.txt
```
notepad.txtで開いているテキストファイルと同じパスで設定ください
## np_image
```
np_image --input [入力画像ファイルのパス]　--output [出力テキストファイルのパス]
# 画像ファイルがinput.png、出力するテキストファイルがout.txtなら
np_image --input input.png --output out.txt
```
### オプション
```
--width 横の文字数を指定できる(デフォルト200)
--color カラー表示ができます。MSゴシックのみ
--brightness 明るさを指定できる。
--help ヘルプが表示できる
```
## np_video
```
np_video --input [入力動画ファイルのパス]　--output [出力テキストファイルのパス]
# 画像ファイルがinput.mp4、出力するテキストファイルがout.txtなら
# inputオプションが整数値の場合、カメラを指定できます。
np_video --input input.mp4 --output out.txt
```
### オプション
```
--width 横の文字数を指定できる(デフォルト200)
--color カラー表示ができます。MSゴシックのみ
--brightness 明るさを指定できる。
--drop-late 遅れたフレームをスキップして再生を滑らかにする。
--help ヘルプが表示できる

np_video --input input.mp4 --output out.txt --width 500 --drop_late 120000 --color
```

# notepad.exeの推奨設定
## フォント
フォントはMSゴシック  
スタイルは標準  
サイズは12  
表示サイズは10%

# np_doom
np_doomはDOOMをnotepad.exeに表示できるスクリプトファイルです。  
np_doomはvisdoomを使用して作成していますのでpython版を実行する場合はvisdoomをインストールしてください。 
またrequire.txtにvisdoomは書かれていません。
```
np_doom --wad [wadファイルパス]　--output [出力テキストファイルのパス]
```
画像ファイルがDOOM.WAD、出力するテキストファイルがout.txtであれば
```
np_doom --wad DOOM.Wad --output out.txt
```
np_doom.pyのオプションはnp_video.pyとほぼ同じです。
操作は、十字キーが「移動」、Ctrlキー又は左クリックが「攻撃」、スペースキーで「使用する」  
ゲーム内で終了をしないと適切に終了できません。
# Qiita
ここにQiita記事