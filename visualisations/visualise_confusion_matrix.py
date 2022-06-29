
def get_confusion_mat_per_epoch(
    pred: tf.Tensor,
    target: tf.Tensor,
    dir: str,
    save: bool = True,
):
    ConfusionMatrixDisplay.from_predictions(
        target, pred)
    
    if save == True:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(
            os.path.join(dir, 'confusion_mat.png')
        )
    return plt.show()