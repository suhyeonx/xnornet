import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

#시각화
def imshow(dataset):
    fig = plt.figure(figsize=(12, 8))
    cols, rows = 8, 5

    # 데이터셋에서 이미지와 레이블 가져오기
    for i in range(1, cols * rows + 1):
        #딕셔너리(dictionary)는 items()함수를 사용하면 딕셔너리에 있는 키와 값들의 쌍을 얻을 수 있습니다. 
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img,label= dataset[sample_idx]
        cls=dataset.classes

        # 이미지를 numpy 배열로 변환하고 형식 변환
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        else:
            img = np.array(img)

        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img)
        ax.set_title(f'Label: {cls[label]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


