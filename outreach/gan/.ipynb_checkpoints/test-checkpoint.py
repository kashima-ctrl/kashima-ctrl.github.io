
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np


# モデルのダウンロード
gan = hub.Module("https://tfhub.dev/google/progan-128/1")


# 64個の 512次元乱数をモデルに入力
z_values = np.random.randn(64, 512)
images = gan(z_values)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 画像生成
    out = sess.run(images)

    # 8行 8列で表示
    r, c = 8, 8
    fig, axs = plt.subplots(r, c, figsize=(10,10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(out[cnt])
            axs[i,j].axis('off')
            cnt += 1

    fig.savefig("images.png")
    plt.show()
    plt.close()
