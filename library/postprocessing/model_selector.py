class ModelSelector:

    @staticmethod
    def select_model_by_lowest_loss(evaluation_data: list) -> dict:
        """
        Returns the model iteration with the lowest loss
        @param evaluation_data:
        @return:
        """
        reconstruction_loss: float = 999999
        selected_fold = {}
        for validation_data in evaluation_data:
            if validation_data["loss"] < reconstruction_loss:
                selected_fold = validation_data
                reconstruction_loss = validation_data["loss"]

        return selected_fold
