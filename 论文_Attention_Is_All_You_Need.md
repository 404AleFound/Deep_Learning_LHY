---

---

## 模型架构

### 编码器和解码器(Encoder and Decoder Stacks)

### 注意力机制(Attention)

```mermaid
%%{init: {
    'theme': 'base',
    'themeVariables': {
        'primaryColor': '#F8F9FA',
        'primaryBorderColor': '#495057',
        'lineColor': '#343A40',
        'textColor': '#212529',
        'fontFamily': '"CMU Serif", "Times New Roman", serif',
        'nodeBorder': '#495057',
        'edgeLabelBackground': '#F8F9FA',
        'fontSize': '16px'
    },
    'config': {
        'flowchart': {
            'curve': 'basis',
            'useMaxWidth': true
        }
    }
}}%%
graph LR
    %% 核心节点
    A[["Attention Mechanism<br>注意力机制"]]:::title
    B[["Query Vector<br>查询向量"]]:::query
    C[["Key-Value Pairs<br>键值对"]]:::kv
    D[["Weighted Output<br>加权输出"]]:::output

    %% 连接关系
    A -->|包含| B
    A -->|包含| C
    A -->|包含| D

    %% 样式类定义
    classDef title fill:#2C3E50,stroke:#1A252F,color:white,font-size:18px,font-weight:bold
    classDef query fill:#E3F2FD,stroke:#1976D2,stroke-width:1.5px
    classDef kv fill:#E8F5E9,stroke:#388E3C,stroke-width:1.5px
    classDef output fill:#FFF3E0,stroke:#FFA000,stroke-width:1.5px
    classDef default rx:4px,ry:4px,min-width:180px

    %% 连接线样式
    linkStyle 0 stroke:#1976D2,stroke-width:2px
    linkStyle 1 stroke:#388E3C,stroke-width:2px
    linkStyle 2 stroke:#FFA000,stroke-width:2px
```

```mermaid
%%{init: {
    'theme': 'base',
    'themeVariables': {
        'primaryColor': '#F8F9FA',
        'primaryBorderColor': '#495057',
        'lineColor': '#343A40',
        'textColor': '#212529',
        'fontFamily': '"CMU Serif", "Times New Roman", serif',
        'nodeBorder': '#495057',
        'edgeLabelBackground': '#F8F9FA',
        'fontSize': '16px'
    },
    'config': {
        'flowchart': {
            'curve': 'basis',
            'useMaxWidth': true
        }
    }
}}%%
graph LR
    %% 节点定义
    B[["Query<br>查询向量"]]:::query
    C[["Key-Value<br>键值对"]]:::kv
    D[["Output<br>加权输出"]]:::output
    E[["Compatibility Function<br>功能函数"]]:::func

    %% 连接关系
    B -->|输入| E
    C -->|输入| E
    E -->|输出| D

    %% 样式类定义
    classDef query fill:#E3F2FD,stroke:#1976D2,stroke-width:1.5px
    classDef kv fill:#E8F5E9,stroke:#388E3C,stroke-width:1.5px
    classDef output fill:#FFF3E0,stroke:#FFA000,stroke-width:1.5px
    classDef func fill:#2C3E50,stroke:#1A252F,color:white,font-weight:bold
    classDef default rx:4px,ry:4px,min-width:160px

    %% 连接线样式
    linkStyle 0 stroke:#1976D2,stroke-width:2px,stroke-dasharray:3
    linkStyle 1 stroke:#388E3C,stroke-width:2px,stroke-dasharray:3
    linkStyle 2 stroke:#FFA000,stroke-width:2px
```

<img src="./assets/image-20250803212758798.png" alt="image-20250803212758798" style="zoom: 33%;" />



<img src="./assets/image-20250803212826635.png" alt="image-20250803212826635" style="zoom:50%;" />



这里的功能函数是用于计算注意力的权重值的，不同的论文有不同的计算方法，最常见的方法有点积和累加这两种。这篇论文在点积的基础上又添加了一些修改，在得到点积值后，将该值除以向量的长度，并使用`softmax`函数加权到0-1之间。

> [!NOTE]
>
> 在得到点积值后，将该值除以向量的长度？


$$
V=[v_1, v_2, v_3,\cdots, v_n]\\
Q=[q_1, q_2, q_3,\cdots, q_n]\\
K=[k_1, k_2, k_3,\cdots, k_n]
$$

$$
Attention(Q,K,V)=softmax(\frac{Q^TK}{\sqrt{(d_k)}})\\
=softmax(
\left[
\begin{array}{c}
q_1k_1 & q_1k_2 & \cdots & q_1k_n\\
q_2k_1 & q_2k_2 & \cdots & q_2k_n\\
\vdots & \vdots & \vdots & \vdots\\
q_nk_1 & q_nk_2 & \cdots & q_nk_n\\
\end{array}
\right])/\sqrt{d_k}
$$



多头的注意力机制(Multi-Head Attention)

> [!NOTE]
>
> 为什么要使用多头？







### Position-wise Feed-Forward Networks



### Embeddings and SoftMax



### Embeddings and SoftMax

## 代码实现