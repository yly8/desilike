import os

from desilike.base import Calculator
from desilike.parameter import VariableCollection


class DESY3Theory(Calculator):

    _is_external = True

    def __init__(self, cosmo=None, ini_file_dir=None, ini_file_name=None, cosmosis_dir=None, cosmosis_param_dict=None, params=None):
        self.cosmo = cosmo
        self._cosmo_names = tuple(p.basename for p in cosmo.params if p.varied) if cosmo is not None else ()
        self.params = {p.name: p for p in VariableCollection(params)}

    def __post_init__(self, cosmo=None, ini_file_dir=None, ini_file_name=None, cosmosis_dir=None, cosmosis_param_dict=None, params=None):
        from cosmosis.runtime.pipeline import LikelihoodPipeline
        self.ini_file = os.path.join(ini_file_dir, ini_file_name)
        self.cosmosis_param_dict = cosmosis_param_dict
        environ_bak = os.environ.copy()
        os.environ['COSMOSIS_STD_DIR'] = cosmosis_dir
        os.environ['INI_FILE_DIR'] = ini_file_dir

        try:
            self.cosmosis_pipe = LikelihoodPipeline(self.ini_file)
            self.cosmosis_data = self.cosmosis_pipe.build_starting_block([])
        finally:
            os.environ.clear()
            os.environ.update(environ_bak)

        if self.cosmosis_param_dict == None:
            #ccoem code copied from https://github.com/cosmodesi/desilike/blob/main/desilike/bindings/cosmosis/factory.py
            self.cosmosis_param_dict = {
                                   #still may need to add some parameters here for non-lcdm models
                                   'H0' : ('cosmological_parameters', 'hubble'),
                                   'h'  : ('cosmological_parameters', 'h0'),
                                   'A_s'  : ('cosmological_parameters', 'A_s'),
                                   'ln10^{10}A_s'  : ('cosmological_parameters', 'log1e10As'),
                                   'sigma8'  : ('cosmological_parameters', 'sigma_8'),
                                   'n_s'  : ('cosmological_parameters', 'n_s'),
                                   'omega_b'  : ('cosmological_parameters', 'ombh2'),
                                   'Omega_b'  : ('cosmological_parameters', 'omega_b'),
                                   'omega_cdm'  : ('cosmological_parameters', 'omch2'),
                                   'Omega_cdm'  : ('cosmological_parameters', 'omega_c'),
                                   'Omega_ncdm'  : ('cosmological_parameters', 'omega_nu'),
                                   'omega_ncdm'  : ('cosmological_parameters', 'omnuh2'),
                                   'm_ncdm'  : ('cosmological_parameters', 'mnu'),
                                   'Omega_k'  : ('cosmological_parameters', 'omega_k'),
                                   'Omega_m'  : ('cosmological_parameters', 'omega_m'),
                                   #now all the photo_z nuisance parameters
                                   'shear_calibration_parameters_m1'  :  ('shear_calibration_parameters', 'm1'),
                                   'shear_calibration_parameters_m2'  :  ('shear_calibration_parameters', 'm2'),
                                   'shear_calibration_parameters_m3'  :  ('shear_calibration_parameters', 'm3'),
                                   'shear_calibration_parameters_m4'  :  ('shear_calibration_parameters', 'm4'),
                                   'wl_photo_z_errors_bias_1'         :  ('wl_photo_z_errors_bias', 'bias_1'),
                                   'wl_photo_z_errors_bias_2'         :  ('wl_photo_z_errors_bias', 'bias_2'),
                                   'wl_photo_z_errors_bias_3'         :  ('wl_photo_z_errors_bias', 'bias_3'),
                                   'wl_photo_z_errors_bias_4'         :  ('wl_photo_z_errors_bias', 'bias_4'),
                                   'lens_photoz_errors_bias_1'        :  ('lens_photoz_errors', 'bias_1'),
                                   'lens_photoz_errors_bias_2'        :  ('lens_photoz_errors', 'bias_2'),
                                   'lens_photoz_errors_bias_3'        :  ('lens_photoz_errors', 'bias_3'),
                                   'lens_photoz_errors_bias_4'        :  ('lens_photoz_errors', 'bias_4'),
                                   'lens_photoz_errors_bias_5'        :  ('lens_photoz_errors', 'bias_5'),
                                   'lens_photoz_errors_bias_6'        :  ('lens_photoz_errors', 'bias_6'),
                                   'lens_photoz_errors_width_1'        :  ('lens_photoz_errors', 'width_1'),
                                   'lens_photoz_errors_width_2'        :  ('lens_photoz_errors', 'width_2'),
                                   'lens_photoz_errors_width_3'        :  ('lens_photoz_errors', 'width_3'),
                                   'lens_photoz_errors_width_4'        :  ('lens_photoz_errors', 'width_4'),
                                   'lens_photoz_errors_width_5'        :  ('lens_photoz_errors', 'width_5'),
                                   'lens_photoz_errors_width_6'        :  ('lens_photoz_errors', 'width_6'),
                                   'bias_lens_b1'        :  ('bias_lens', 'b1'),
                                   'bias_lens_b2'        :  ('bias_lens', 'b2'),
                                   'bias_lens_b3'        :  ('bias_lens', 'b3'),
                                   'bias_lens_b4'        :  ('bias_lens', 'b4'),
                                   'bias_lens_b5'        :  ('bias_lens', 'b5'),
                                   'bias_lens_b6'        :  ('bias_lens', 'b6'),
                                   'mag_alpha_lens_mag_alpha_lens_1'        :  ('mag_alpha_lens', 'mag_alpha_lens_1'),
                                   'mag_alpha_lens_mag_alpha_lens_2'        :  ('mag_alpha_lens', 'mag_alpha_lens_2'),
                                   'mag_alpha_lens_mag_alpha_lens_3'        :  ('mag_alpha_lens', 'mag_alpha_lens_3'),
                                   'mag_alpha_lens_mag_alpha_lens_4'        :  ('mag_alpha_lens', 'mag_alpha_lens_4'),
                                   'mag_alpha_lens_mag_alpha_lens_5'        :  ('mag_alpha_lens', 'mag_alpha_lens_5'),
                                   'mag_alpha_lens_mag_alpha_lens_6'        :  ('mag_alpha_lens', 'mag_alpha_lens_6'),
                                   'intrinsic_alignment_parameters_z_piv'   :  ('intrinsic_alignment_parameters' , 'z_piv'),
                                   'intrinsic_alignment_parameters_a1'   :  ('intrinsic_alignment_parameters' , 'a1'),
                                   'intrinsic_alignment_parameters_alpha1'   :  ('intrinsic_alignment_parameters' , 'alpha1'),
                                   }
        self.cosmosis_param_dict.setdefault('logA', ('cosmological_parameters', 'log1e10As'))
        if self.cosmo is not None:
            self.cosmo.add_requirements({f'params.{name}': None for name in self._cosmo_names})

    def __call__(self):
        # Cosmology values are explicit graph leaves, including inside callbacks.
        for name in self._cosmo_names:
            self.cosmosis_data[self.cosmosis_param_dict[name]] = float(self.cosmo[name])
        # do the translation
        for key, param in self.params.items():
            self.cosmosis_data[self.cosmosis_param_dict[key]] = float(param.value)

        # run cosmosis
        self.cosmosis_pipe.run(self.cosmosis_data)
        self.theory_vector = self.cosmosis_data['data_vector', '2pt_theory']

        return self.theory_vector

    def tree_flatten(self):
        return [self.theory_vector], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.theory_vector = children[0]
        return obj
