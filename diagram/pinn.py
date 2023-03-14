
import numpy as np
import sys
sys.path.append('../')
from pycore.tikzeng import *

offset = -2
scale = .5
decoder_offset = 0
def ppos(num_tuple_s):
    nums = num_tuple_s.replace(' ', '').replace('(', '').replace(')', '')
    nums = nums.split(',')
    nums = np.array([float(n) for n in nums])
    nums[0] += decoder_offset
    new = str(tuple(n * scale for n in nums))
    print(new)
    return new

def ppos_encoder(num_tuple_s):
    nums = num_tuple_s.replace(' ', '').replace('(', '').replace(')', '')
    nums = nums.split(',')
    new = str(tuple(float(n) * scale for n in nums))
    print(new)
    return new

## vim macro
"""/\dcw{}jk?codek$a"jkpa", jk/\a\d"""

zoff = -10 #* scale
xext = 31
xpatch = 31.2
patch_width = .5
diff_width = .5
diff_spacing = 7
xdiff = 37
xdiff2 = 44

diff2_spacing = .2
diff2_dx = 2
diff2_dy = .5
diff2_dz = 0.5
diff2_width = 3
amp_suffix = '_1'
phase_suffix = '_2'

legend_offset_y = -15
legend_boxsize = 8
legend_width = 0
legend_spacing_x = 6
legend_spacing_y = -6
legend_patch_width = .1
offset2 = offset * (legend_boxsize / 32)

img_path_fmt = '../../notebooks/images/{}'

input1 = img_path_fmt.format('in1.jpeg')
input2 = img_path_fmt.format('in2.jpeg')
input3 = img_path_fmt.format('in3.jpeg')
input4 = img_path_fmt.format('in4.jpeg')

output1 = img_path_fmt.format('out1.jpeg')
output2 = img_path_fmt.format('out2.jpeg')
output3 = img_path_fmt.format('out3.jpeg')
output4 = img_path_fmt.format('out4.jpeg')

patch1_path = img_path_fmt.format('patch1.jpeg')
patch2_path = img_path_fmt.format('patch2.jpeg')
patch3_path = img_path_fmt.format('patch3.jpeg')
patch4_path = img_path_fmt.format('patch4.jpeg')

phase_path = img_path_fmt.format('phase.jpeg')
amp_path = img_path_fmt.format('amp.jpeg')
full_obj_path = img_path_fmt.format('full_obj.jpeg')

im_size = 6.5
inp_x = -7
outp_x = 50

encoder = [
    to_input(input1, to=ppos("({},{},{})".format(inp_x, diff2_dy * 1.5, diff2_dz * 1.5)), width = im_size, height = im_size),
    to_input(input2, to=ppos("({},{},{})".format(inp_x, diff2_dy * .5, diff2_dz * .5)), width = im_size, height = im_size),
    to_input(input3, to=ppos("({},{},{})".format(inp_x, diff2_dy * -.5, diff2_dz * -.5)), width = im_size, height = im_size),
    to_input(input4, to=ppos("({},{},{})".format(inp_x, diff2_dy * -1.5, diff2_dz * -1.5)), width = im_size, height = im_size),
    to_ConvRelu("conv11", '', 64, offset=ppos_encoder("(0,0,0)"), to=ppos_encoder("(0,0,0)"),
        height=64 * scale, depth=64 * scale, width=2 * scale),
    to_ConvRelu("conv12", '', '', offset=ppos_encoder("(.4,0,0)"), to=ppos_encoder("(0,0,0)"), height=64 * scale, depth=64 * scale, width=2 * scale),
    to_Pool("pool1", offset=ppos_encoder("(0,0,0)"), to="(conv12-east)", height=32* scale, depth=32* scale),

    to_ConvRelu("conv21", '', '', offset=ppos_encoder("(5,0,0)"), to=ppos_encoder("(0,0,0)"), height=32* scale, depth=32* scale, width=4 * scale),
    to_ConvRelu("conv22", '', 128, offset=ppos_encoder("(5.8,0,0)"), to=ppos_encoder("(0,0,0)"), height=32* scale, depth=32* scale, width=4 * scale),
    to_Pool("pool2", offset=ppos_encoder("(0,0,0)"), to="(conv22-east)", height=16* scale, depth=16* scale),
    to_connection( "pool1", "conv21"),

    to_ConvRelu("conv31", '', 256, offset=ppos_encoder("(10,0,0)"), to=ppos_encoder("(0,0,0)"), height=16* scale, depth=16* scale, width=8 * scale),
    to_ConvRelu("conv32", '', '', offset=ppos_encoder("(11.6,0,0)"), to=ppos_encoder("(0,0,0)"), height=16* scale, depth=16* scale, width=8 * scale),
    to_Pool("pool3", offset=ppos_encoder("(0,0,0)"), to="(conv32-east)", height=8* scale, depth=8* scale),
    to_connection( "pool2", "conv31"),
]

def last_decoder(pos_sign, name_suffix):
    if pos_sign == 1:
        return to_Sigmoid("last" + name_suffix, '', '', offset=ppos("(23.2,0,0)"),
            to=ppos("(0,{},0)".format(pos_sign * zoff)), height=32* scale,
            depth=32* scale, width=2 * scale)
    elif pos_sign == -1:
        return to_Tanh("last" + name_suffix, '', '', offset=ppos("(23.2,0,0)"),
            to=ppos("(0,{},0)".format(pos_sign * zoff)), height=32* scale,
            depth=32* scale, width=2 * scale)
    else:
        raise ValueError

def last_decoder_img(pos_sign, name_suffix):
    if pos_sign == 1:
        return to_input(amp_path,
            to=ppos("(24.2,{},0)".format(pos_sign * zoff)), width = im_size / 2,
            height = im_size / 2)
    elif pos_sign == -1:
        return to_input(phase_path,
            to=ppos("(24.2,{},0)".format(pos_sign * zoff)), width = im_size / 2,
            height = im_size / 2)
    else:
        raise ValueError

def mk_decoder(name_suffix = '0', pos_sign = 1):
    return [
    to_ConvRelu("up11" + name_suffix, '', 256, offset=ppos("(12,0,0)"), to=ppos("(0,{},0)".format(pos_sign * zoff)), height=8* scale, depth=8* scale, width=8 * scale),
    to_ConvRelu("up12" + name_suffix, '', '', offset=ppos("(13.6,0,0)"), to=ppos("(0,{},0)".format(pos_sign * zoff)), height=8* scale, depth=8* scale, width=8 * scale),
    to_UnPool("unpool1" + name_suffix, offset=ppos("(0,0,0)"), to="(up12" + name_suffix + "-east)", height=16* scale, depth=16* scale),
    to_connection( "pool3", "up11" + name_suffix),

    to_ConvRelu("up21" + name_suffix, '', 128, offset=ppos("(17,0,0)"), to=ppos("(0,{},0)".format(pos_sign * zoff)), height=16* scale, depth=16* scale, width=4 * scale),
    to_ConvRelu("up22" + name_suffix, '', '', offset=ppos("(17.8,0,0)"), to=ppos("(0,{},0)".format(pos_sign * zoff)), height=16* scale, depth=16* scale, width=4 * scale),
    to_UnPool("unpool2" + name_suffix, offset=ppos("(0,0,0)"), to="(up22" + name_suffix + "-east)", height=32* scale, depth=32* scale),
    to_connection( "unpool1" + name_suffix, "up21" + name_suffix),

#    to_Conv("up31" + name_suffix, '', 1, offset=ppos("(23,0,0)"),
#        to=ppos("(0,{},0)".format(pos_sign * zoff)), height=32* scale,
#        depth=32* scale, width=2 * scale),
    last_decoder(pos_sign, name_suffix),
    last_decoder_img(pos_sign, name_suffix),
    to_connection( "unpool2" + name_suffix, last + name_suffix)
    #to_connection( "unpool2" + name_suffix, "up31" + name_suffix)
    ]
#last = "up31"
last = "last"

forward_map =\
[
    to_Sum("sum1", offset=ppos("(27.5,0,0)"), to=ppos("(0, 0, 0)"), radius=2.5 * scale, opacity=0.6),
    to_connection(last+ amp_suffix, "sum1"),
    to_connection(last+ phase_suffix, "sum1"),
    to_Extract("extract1", '', 4, offset=ppos("({},0,0)".format(xext)),
        to=ppos("(0,0,0)"), height=64* scale / 2, depth=64* scale / 2, width=2* scale,
        caption = 'Unstack offsets'),
    to_input(full_obj_path, to = ppos("({},0,0)".format(xext)), width = im_size,
        height = im_size),
    to_connection("sum1", "extract1"),
    to_Patch("patch1", '', '', offset=ppos("({},0,0)".format(xpatch)),
        to=ppos("(0,{},{})".format(offset, offset)), height=32* scale, depth=32* scale,
        width=patch_width * scale),
    to_Patch("patch4", '', '', offset=ppos("({},0,0)".format(xpatch)),
        to=ppos("(0,{},{})".format(offset, -offset)), height=32* scale, depth=32* scale,
        width=patch_width * scale),
    to_Patch("patch2", '', '', offset=ppos("({},0,0)".format(xpatch)),
        to=ppos("(0,{},{})".format(-offset, offset)), height=32* scale, depth=32* scale,
        width=patch_width * scale),
    to_Patch("patch3", '', '', offset=ppos("({},0,0)".format(xpatch)),
        to=ppos("(0,{},{})".format(-offset, -offset)), height=32* scale, depth=32* scale,
        width=patch_width * scale)] +\
    to_Illumination("probe2", patch2_path, '', '', offset=ppos("(0,0,0)"),
        to=ppos("({},{},0)".format(xdiff, diff_spacing * 1.5)), height=32* scale, depth=32* scale,
        width=diff_width * scale, im_size = im_size / 2) +\
    [to_connection( "patch2", "probe2")] +\
    to_Illumination("probe3", patch3_path, '', '', offset=ppos("(0,0,0)"),
        to=ppos("({},{},0)".format(xdiff, diff_spacing * .5)), height=32* scale, depth=32* scale,
        width=diff_width * scale, im_size = im_size / 2) +\
    [to_connection( "patch3", "probe3")] +\
    to_Illumination("probe1", patch1_path, '', '', offset=ppos("(0,0,0)"),
        to=ppos("({},{},0)".format(xdiff, diff_spacing * -.5)), height=32* scale, depth=32* scale,
        width=diff_width * scale, im_size = im_size / 2) +\
    [to_connection( "patch1", "probe1")] +\
    to_Illumination("probe4", patch4_path, '', '', offset=ppos("(0,0,0)"),
        to=ppos("({},{},0)".format(xdiff, diff_spacing * -1.5)), height=32* scale, depth=32* scale,
        width=diff_width * scale, im_size = im_size / 2) +\
    [to_connection( "patch4", "probe4")] +\
    [
#        to_Diffraction("diff1", '', 4, offset=ppos("({},0,0)".format(xdiff2)),
#        to=ppos("({},{},{})".format(diff2_dx * 1.5, diff2_dy * 1.5, diff2_dz * 1.5)), height=64* scale, depth=64* scale, width=diff2_width* scale,
#        caption = ''),
#    to_Diffraction("diff2", '', 4, offset=ppos("({},0,0)".format(xdiff2 + diff2_spacing)),
#        to=ppos("({},{},{})".format(diff2_dx * .5, diff2_dy * .5, diff2_dz * .5)), height=64* scale, depth=64* scale, width=diff2_width* scale,
#        caption = ''),
    to_Diffraction("diff3", '', 4, offset=ppos("({},0,0)".format(xdiff2 + diff2_spacing * 2)),
        to=ppos("({},{},{})".format(diff2_dx * -.5, diff2_dy * -.5, diff2_dz * -.5)), height=64* scale, depth=64* scale, width=diff2_width* scale,
        caption = ''),
#    to_Diffraction("diff4", '', 4, offset=ppos("({},0,0)".format(xdiff2 + diff2_spacing * 3)),
#        to=ppos("({},{},{})".format(diff2_dx * -1.5, diff2_dy * -1.5, diff2_dz * -1.5)), height=64* scale, depth=64* scale, width=diff2_width* scale,
#        caption = ''),
    to_connection("probe2", "diff3"),#top
    to_connection("probe1", "diff3"),
    to_connection("probe3", "diff3"),
    to_connection("probe4", "diff3"),# bottom
#    to_connection("probe2", "diff1"),#top
#    to_connection("probe1", "diff3"),
#    to_connection("probe3", "diff2"),
#    to_connection("probe4", "diff4"),# bottom

    to_input(output1, to=ppos("({},{},{})".format(outp_x, diff2_dy * 1.5, diff2_dz * 1.5)), width = im_size, height = im_size),
    to_input(output2, to=ppos("({},{},{})".format(outp_x, diff2_dy * .5, diff2_dz * .5)), width = im_size, height = im_size),
    to_input(output3, to=ppos("({},{},{})".format(outp_x, diff2_dy * -.5, diff2_dz * -.5)), width = im_size, height = im_size),
    to_input(output4, to=ppos("({},{},{})".format(outp_x, diff2_dy * -1.5, diff2_dz * -1.5)), width = im_size, height = im_size),

    to_ConvRelu("conv_relu_legend", offset=ppos_encoder("(0,{},0)".format(legend_offset_y)),
        to=ppos_encoder("(0,0,0)"), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_width * scale,
        caption = r"""Conv2D($\cdot$)$\linebreak$~ReLU($\cdot$)""",
        #caption = r"""Conv2D($\cdot$)$\linebreak\rightarrow$~ReLU($\cdot$)""",
        s_filer = '', n_filer = ''),

    to_Pool("pool_legend", offset=ppos_encoder("({},{},0)".format(legend_spacing_x, legend_offset_y)),
        to=ppos_encoder("(0,0,0)"), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_width * scale,
        caption = r"""AvgPool2D($\cdot$)"""),

    to_UnPool("unpool_legend", offset=ppos_encoder("({},{},0)".format(legend_spacing_x * 2,
        legend_offset_y)),
        to=ppos_encoder("(0,0,0)"), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_width * scale,
        caption = r"""Upsample($\cdot$)"""),

    to_Tanh("tanh_legend", offset=ppos_encoder("({},{},0)".format(legend_spacing_x * 0,
            legend_offset_y + legend_spacing_y)),
        to=ppos_encoder("(0,0,0)"), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_width * scale,
        caption = r"""Conv2D($\cdot$)$\linebreak$~$i \bm{\pi \tanh(\cdot)}$""",
        #caption = r"""Conv2D($\cdot$)$\linebreak\rightarrow$~$i \pi \tanh(\cdot$)""",
        s_filer = '', n_filer = ''),

    to_Sigmoid("sigmoid_legend", offset=ppos_encoder("({},{},0)".format(legend_spacing_x * 1,
            legend_offset_y + legend_spacing_y)),
        to=ppos_encoder("(0,0,0)"), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_width * scale,
        caption = r"""Conv2D($\cdot$)$\linebreak$~Sigmoid$(\cdot)$""",
        s_filer = '', n_filer = ''),

    to_Patch("patch1_legend", '', '', offset=ppos("({},{},0)".format(legend_spacing_x * 2,
            legend_offset_y + legend_spacing_y)),
        to=ppos("(0,{},{})".format(offset2, offset2)), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_patch_width * scale),
    to_Patch("patch4_legend", '', '', offset=ppos("({},{},0)".format(legend_spacing_x * 2,
            legend_offset_y + legend_spacing_y)),
        to=ppos("(0,{},{})".format(offset2, -offset2)), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_patch_width * scale),
    to_Patch("patch2_legend", '', '', offset=ppos("({},{},0)".format(legend_spacing_x * 2,
            legend_offset_y + legend_spacing_y)),
        to=ppos("(0,{},{})".format(-offset2, offset2)), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_patch_width * scale),
    to_Patch("patch3_legend", '', '', offset=ppos("({},{},0)".format(legend_spacing_x * 2,
            legend_offset_y + legend_spacing_y)),
        to=ppos("(0,{},{})".format(-offset2, -offset2)), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_patch_width * scale,
        caption = r"""Illuminate$(\cdot)$""",),

    to_Diffraction("diff_legend", offset=ppos_encoder("({},{},0)".format(legend_spacing_x * 3,
            legend_offset_y + legend_spacing_y)),
        to=ppos_encoder("(0,0,0)"), height=legend_boxsize * scale, depth=legend_boxsize * scale,
        width=legend_width * scale,
        caption = r"""Diffract$(\cdot)$""",
        s_filer = '', n_filer = ''),

]

arch = [to_head( '..' ),
    to_cor(),
    to_begin()] +\
    encoder + mk_decoder(amp_suffix, pos_sign = 1) +\
    mk_decoder(phase_suffix, pos_sign = -1) + forward_map +\
    [to_end()]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
