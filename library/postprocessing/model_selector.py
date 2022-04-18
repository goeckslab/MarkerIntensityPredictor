class ModelSelector:

    @staticmethod
    def select_model_by_lowest_loss(evaluation_data: dict) -> dict:
        """
        Returns the model iteration with the lowest loss
        @param evaluation_data:
        @return:
        """
        validation_mean_loss: float = 999999
        selected_fold = {}
        for model, validation_list in evaluation_data.items():
            mean_loss = 0
            for validation_data in validation_list:
                mean_loss += validation_data["loss"]

            mean_loss = mean_loss / len(validation_list)

            if mean_loss < validation_mean_loss:
                selected_fold = validation_list[0]
                validation_mean_loss = mean_loss

        return selected_fold
