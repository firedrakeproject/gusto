
sw_model = Model()
ml_model = PointNN(nn_in=5, nn_out=3)
hybrid_model = HybridModel(sw_model, ml_model, fields_to_adjust=["u", "D"])
hybrid_model.generate_data(filename)
hybrid_model.train()
hybrid_model.evaluate()
