class Stats:
    @staticmethod
    def get_percentage(new_num, org_num):
        return round((new_num * 100) / org_num, 2)

    @staticmethod
    def print_data_overview(init_inputs, inputs, X_train, init_X_train, init_X_test, init_X_val, X_test, X_val):
        print(
            f"              Input Shape:\t{init_inputs.shape}\t[{init_inputs.shape[0]}] Cells and [{init_inputs.shape[1]}] Markers\n")
        print(f"                Input size before normalization:\t{init_inputs.shape}")
        print(f"                Input size after  normalization:\t{inputs.shape}")
        print(
            f"                Removed data percentage:\t{100 - Stats.get_percentage(inputs.shape[0], init_inputs.shape[0])}%")
        print("")
        print(
            f"              Train data size:\t{X_train.shape}\t({Stats.get_percentage(init_X_train.shape[0], init_inputs.shape[0])}% of input)")
        print(
            f"              Test data size:\t{X_test.shape}\t({Stats.get_percentage(init_X_test.shape[0], init_inputs.shape[0])}% of input)")
        print(
            f"              Validation data size:\t{X_val.shape}\t({Stats.get_percentage(init_X_val.shape[0], init_inputs.shape[0])}% of input)")
