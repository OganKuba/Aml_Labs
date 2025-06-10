import matplotlib.pyplot as plt

def plot_body_surface(df):
    class0 = df[df['popn'] == 0]
    class1 = df[df['popn'] == 1]

    plt.figure()
    plt.scatter(class0['body'], class0['surface'], label='Class 0', marker='o')
    plt.scatter(class1['body'], class1['surface'], label='Class 1', marker='x')
    plt.xlabel('body')
    plt.ylabel('surface')
    plt.title('Scatterplot: body vs. surface')
    plt.legend()
    plt.show()