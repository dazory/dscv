

class ParamManager():
    def __init__(self):
        super(ParamManager, self).__init__()

    # Debuging tools
    def check_param_structure(self, model, blank=''):
        '''
        Example: check_param_structure(model)
        '''
        for name, child in model.named_children():
            print(f"{blank}{name}: ")
            self.check_param_structure(child, blank + '  ├ ')

    def check_grad_status(self, model, blank=''):
        '''
        Example: check_grad_status(model)
        '''
        for name, child in model.named_children():
            print(f"{blank}{name}: ", end='')
            for param in child.parameters():
                print(f"{param.requires_grad} ", end='')
            print('')
            self.check_grad_status(child, blank + '  ├ ')

    # Helper
    def find_layer(self, model, layer_name):
        '''
        Input:
                layer_name = (str)
        Example:
                layer_name = 'rpn.rpn_cls_layer'
                layer = find_layer(model, layer_name)
                print(layer)
        '''
        layer_name = layer_name.split('.')
        if len(layer_name) != 0:
            for name, child in model.named_children():
                if name == layer_name[0]:
                    if len(layer_name) == 1:
                        return child
                    child = self.find_layer(child, layer_name[1:])
                    if child != False:
                        return child
        return False

    # Freeze
    def freeze_all_params(self, model):
        '''
        Example: freeze_all_params(model)
        '''
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_all_params(child)

    def freeze_params(self, model, layer_to_freeze_list):
        '''
        Output:
                freezed_layer_list = True if all layer in layer_to_freeze_list are freezed.
                                                         (list) else the list of freezed layers.
        Example:
                layer_to_freeze_list = ['rpn.rpn_cls_layer', 'rpn.rpn_reg_layer', 'rcnn_net.cls_layer', 'rcnn_net.reg_layer']
                print(freeze_params(model, layer_to_freeze_list))
        '''
        freezed_layer_list = []
        for layer_to_freeze in layer_to_freeze_list:
            layer = self.find_layer(model, layer_to_freeze)
            if layer != False:
                self.freeze_all_params(layer)
                freezed_layer_list.append(layer_to_freeze)

        return True if len(layer_to_freeze_list) == len(freezed_layer_list) \
            else freezed_layer_list

