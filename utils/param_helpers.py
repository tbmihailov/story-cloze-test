
def get_settings_dict(settings_str, delim_main="|", delim_kv="="):
    """
    Get key-values from string Ex. "a=0.6|b=9|0.4" => {"a":'0.6', "b":'9', 2:"0.4"} - 0.4 has key the index in the setting.
    :param settings_str: Settings string. Ex. "a=0.6|b=9|0.4"
    :param delim_main: Main delimiter. Ex "|"
    :param delim_kv: key-value delim. Ex. "="
    :return:
    """
    settings_list = settings_str.split(delim_main)
    settings_list_tuple = [x.strip().split(delim_kv) for x in settings_list]
    settings_list_tuple = [[x1.strip() for x1 in x] for x in settings_list_tuple]

    settings_dict = {x[0] if len(x)>1 else i: x[1] if len(x)>1 else x[0] for i,x in enumerate(settings_list_tuple)}

    return settings_dict