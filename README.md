# notepad_rendering
windowsのメモ帳、notepad.exeで画像・動画を表示できる。  
新notepad.exeでも表示可能だが、旧notepad.exeのほうがきれいに表示できる。  
テキストを10%表示にすると、文字が2ドットになってアンチエイリアスによって色がつく仕様を使った

#　使い方  
まず表示するためにテキストファイルを作成する。  
次にテキストファイルをnotepad.exeで開く。  
`np-render.py`をpythonで実行する  
別ウィンドウで画像を表示するなら`np-image.py`、動画を再生するなら`np-video.py`をpythonで実行。  

`--help -h`ヘルプを表示  
`--input`表示したい画像・動画ファイル  
`--output`更新するテキストファイル  
`--width`横の文字数(ドット数)  
`--color`カラー表示  

`np-render.py`はテキストファイルが変更時メモ帳を更新するプログラム。
# notepad.exeの設定
## フォント
フォントはMSゴシック  
スタイルは標準  
サイズは12  
## 表示サイズ
10%

# 参考
https://gigazine.net/news/20221012-doom-notepad/