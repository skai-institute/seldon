from TAE.util import read_transmission_filter
from TAE.util import register_module
from scipy.interpolate import interp1d

legacy_band_index_map = {
    "g": 0, 
    "r": 1, 
    "i": 2, 
    "z": 3, 
    "y": 4, 
    "u": 5,
    "F062": 6,
    "F087": 7,
    "F106": 8,
    "F129": 9,
    "F146": 10,
    "F158": 11,
    "F184": 12,
    "F213": 13,
    "F070W" : 14,
    "F090W" : 15, 
    "F115W" : 16,
    "F140M" : 17, 
    "F150W" : 18,
    "F150W2" : 19,
    "F162M" : 20,
    "F164N" : 21,
    "F182M" : 22,
    "F187N" : 23,
    "F200W" : 24,
    "F210M" : 25,
    "F212N" : 26,
    "F250M" : 27,
    "F277W" : 28,
    "F300M" : 29,
    "F322W2" : 30,
    "F323N" : 31,
    "F335M" : 32,
    "F356W" : 33,
    "F360M" : 34, 
    "F405N" : 35,
    "F410M" : 36,
    "F430M" : 37,
    "F444W" : 38,
    "F460M" : 39,
    "F466N" : 40,
    "F470N" : 41,
    "F480M" : 42,

}





class BandPass(dict):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.index_to_filter = None
        self.num_bands = len(self)
        self.unit = 'microns'

    @classmethod
    def from_sncosmo(cls):

        import sncosmo
        from sncosmo.bandpasses import _BANDPASSES as bandpasses
        primary_filter_keys = [key for key in sorted(bandpasses._primary_loaders, key=lambda x: x[0])]
        loader_to_key = {id(bandpasses._loaders[key]): key for key in primary_filter_keys}

        alias_keys = [i for i in bandpasses._loaders.keys() if i not in primary_filter_keys]
        alias_to_key = {key[0]: loader_to_key[id(bandpasses._loaders[key])][0] for key in alias_keys}
        primary_key_names = sorted([name for name, version in primary_filter_keys])
        band_to_index = dict(zip(*list(zip(*enumerate(primary_key_names)))[::-1]))
        band_to_index.update({key:band_to_index[alias_to_key[key]] for key in alias_to_key})
        num_bands = len(primary_key_names)
        index_to_filter = {band_to_index[key]: sncosmo.get_bandpass(key) for key in primary_key_names}
        index_to_filter = {key: (value.wave, value.trans) for key, value in index_to_filter.items()}

        new_bandpass = cls(**band_to_index)
        new_bandpass.num_bands = num_bands
        new_bandpass.index_to_filter = index_to_filter
        new_bandpass.unit = 'angstroms'

        return new_bandpass

    @classmethod
    def from_files(cls, filedict):
        '''filedict is a map of the form {name:filename}'''
        
        num_bands = len(filedict)
        keys = sorted(list(filedict.keys()))
        band_to_index = dict(zip(*list(zip(*enumerate(keys)))[::-1]))
        index_to_filter = {band_to_index[key]: read_transmission_filter(filedict[key]) for key in keys}
        #index_to_filter = {band_to_index[key]: read_transmission_filter(value) for key, value in filedict.items()}

        new_bandpass = cls(**band_to_index)
        new_bandpass.num_bands = num_bands
        new_bandpass.index_to_filter = index_to_filter
        new_bandpass.unit = 'microns'
        

        return new_bandpass

    @classmethod
    def from_strings(cls, strings):
        '''construct the mapping from a list of strings'''

        strings = sorted(strings)
        band_to_index = dict(zip(*list(zip(*enumerate(strings)))[::-1]))

        new_bandpass = cls(**band_to_index)
        new_bandpass.num_bands = len(strings)

        return new_bandpass

@register_module
def BandIndexMap(mode=None, *args, **kwargs):
    "Modes can be None, 'sncosmo', 'strings', 'files', 'legacy'"

    if mode == None:
        return BandPass(*args, **kwargs)
    elif mode == 'legacy':
        return BandPass(**legacy_band_index_map)

    return getattr(BandPass, "from_"+mode)(*args, **kwargs)

BAND_IDX_MAP = BandIndexMap(mode='legacy')