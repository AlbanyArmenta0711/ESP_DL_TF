
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc  model.cc
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS "")
