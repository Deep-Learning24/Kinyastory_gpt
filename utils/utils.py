def load_weight(model,state_dict):
        model.load_state_dict(state_dict)
        return model