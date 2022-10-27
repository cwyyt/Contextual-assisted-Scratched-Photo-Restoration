from Restoration import Restoration
def creat_model(opt):
    model = Restoration(opt)
    print("model [%s] was created" % (model.name()))
    print(model)
    return model
