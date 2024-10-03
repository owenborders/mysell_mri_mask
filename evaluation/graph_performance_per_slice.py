from matplotlib import pyplot as plt


import nibabel as nib
import numpy as np
import os

class AnalyzePerformance():

	def __init__(self):
		self.dice_per_slice = {'axial': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.6603773460306162, 22: 0.3421052609072023, 23: 1.2200435010767854, 24: 0.29156839468035056, 25: 3.54411086406877, 26: 5.447401631637093, 27: 6.8129215516888255, 28: 9.053748941851609, 29: 8.177328155784124, 30: 17.137281230576043, 31: 15.752915611160011, 32: 13.693626564704653, 33: 12.374417369825942, 34: 12.212330768560268, 35: 15.056304026952985, 36: 15.523965517094465, 37: 9.490875836269797, 38: 7.184523626203166, 39: 7.135087325543066, 40: 6.897314264237825, 41: 6.22509168203317, 42: 4.650207645325628, 43: 4.027640333238795, 44: 3.441284704922728, 45: 2.890369963587582, 46: 2.577991961203682, 47: 2.338337639565872, 48: 2.0888175458475837, 49: 1.9243153671270128, 50: 1.786966482397132, 51: 1.7157861963516927, 52: 1.636799863514848, 53: 1.5510740822693538, 54: 1.4816051307311646, 55: 1.434220420592193, 56: 1.3718058654821879, 57: 1.2973538399574873, 58: 1.2396304937233582, 59: 1.1964122398970152, 60: 1.1667892248958716, 61: 1.1165927199568824, 62: 1.0819641812323744, 63: 1.0545401632755376, 64: 1.0350991484850403, 65: 1.012083016558881, 66: 0.9818409638460145, 67: 0.965354498453526, 68: 0.943954527014815, 69: 0.9228720150635971, 70: 0.9069230395034491, 71: 0.8851560879723082, 72: 0.8649988429375834, 73: 0.8583489625846995, 74: 0.8493776633545366, 75: 0.842787402911307, 76: 0.8361566671886056, 77: 0.8240995647755126, 78: 0.8187198641896328, 79: 0.824248673578371, 80: 0.8157383388314817, 81: 0.8106758952242514, 82: 0.8052704904263008, 83: 0.8016471464451016, 84: 0.8042000593455649, 85: 0.8045716570213726, 86: 0.7880640767192924, 87: 0.7895247874101178, 88: 0.8003365838816413, 89: 0.8054842108482038, 90: 0.809998160456352, 91: 0.8103504033533836, 92: 0.8166448165157507, 93: 0.8214301165569504, 94: 0.8170846157111049, 95: 0.8376780858820919, 96: 0.8495189099537584, 97: 0.8615581979738868, 98: 0.8769562840320722, 99: 0.905222147279176, 100: 0.9332055087280449, 101: 0.9629781506044124, 102: 1.0047048680960016, 103: 1.0455197192597105, 104: 1.0777187046932117, 105: 1.1274435053036491, 106: 1.1625781305013423, 107: 1.1930348826256902, 108: 1.2392236399721974, 109: 1.2907311079287087, 110: 1.336242914879515, 111: 1.3843131753074231, 112: 1.4221952852708493, 113: 1.4931309187560755, 114: 1.5536656968621312, 115: 1.5961992841795207, 116: 1.6557293200774874, 117: 1.7472059514816802, 118: 1.8287133943197955, 119: 1.887365774929581, 120: 1.9695346294617906, 121: 2.023861020613813, 122: 2.085380129555496, 123: 2.1732940456168026, 124: 2.269907289865478, 125: 2.3569232648699563, 126: 2.449980584532123, 127: 2.498532051252535, 128: 2.5056033497074575, 129: 2.583798074653743, 130: 2.654965151378513, 131: 2.68193241904661, 132: 2.7561416792840356, 133: 2.8159119706124875, 134: 2.9561968742711535, 135: 3.0170013432152523, 136: 3.0550529480655797, 137: 2.9394494590227875, 138: 2.937492918841324, 139: 2.9396009260769973, 140: 2.9242705227732206, 141: 3.00257407329826, 142: 2.948359972540239, 143: 2.790386723079693, 144: 2.701582428051002, 145: 2.6950120109965545, 146: 2.754998650680309, 147: 2.6752201900999677, 148: 2.741471008222714, 149: 2.7699774929396117, 150: 2.854728191667569, 151: 3.2224014721686247, 152: 3.3812787454924313, 153: 3.663337972926346, 154: 4.375400896297997, 155: 5.215539156201268, 156: 4.949804861037266, 157: 6.871660934708545, 158: 4.91918189211046, 159: 5.721184302859892, 160: 8.49900621707718, 161: 12.616315949385495, 162: 16.436640572481554, 163: 12.801456838686034, 164: 12.661032307009435, 165: 16.506961112452547, 166: 19.51513894873787, 167: 24.52602674687537, 168: 27.29801971966586, 169: 23.216370941926982, 170: 21.41960832014585, 171: 17.258002372559023, 172: 20.28509870680961, 173: 23.498693789399464, 174: 26.895740051739963, 175: 25.319448523226228, 176: 23.678107554297398, 177: 20.918910406820714, 178: 19.03715054236672, 179: 15.091388319357183, 180: 10.338088018549804, 181: 9.606384213434328, 182: 10.62437743475741, 183: 8.341843509854115, 184: 6.2601614400136025, 185: 7.461154231726712, 186: 6.459438799201607, 187: 6.259681864553986, 188: 3.335087695734847, 189: 3.465346499562813, 190: 2.9999999781480797, 191: 2.999999973792007, 192: 1.9999999156142425, 193: 1.9999998675373292, 194: 0.9999999915254238, 195: 1.9999998964326897, 196: 2.9999998826090835, 197: 1.999999939734836, 198: 1.9999996589148399, 199: 0.9999999923076923, 200: 0.9999999925925926, 201: 0.9999999921875, 202: 0.9999999761904768, 203: 0.0, 204: 0.0, 205: 0.0, 206: 0.0, 207: 0.0, 208: 0.0, 209: 0.0, 210: 0.0, 211: 0.0, 212: 0.0, 213: 0.0, 214: 0.0, 215: 0.0, 216: 0.0, 217: 0.0, 218: 0.0, 219: 0.0, 220: 0.0, 221: 0.0, 222: 0.0, 223: 0.0, 224: 0.0, 225: 0.0, 226: 0.0, 227: 0.0, 228: 0.0, 229: 0.0, 230: 0.0, 231: 0.0, 232: 0.0, 233: 0.0, 234: 0.0, 235: 0.0, 236: 0.0, 237: 0.0, 238: 0.0, 239: 0.0, 240: 0.0, 241: 0.0, 242: 0.0, 243: 0.0, 244: 0.9999999887640451, 245: 0.0, 246: 0.9999999866666669, 247: 0.0, 248: 0.0, 249: 0.0, 250: 0.0, 251: 0.0, 252: 0.0, 253: 0.0, 254: 0.0, 255: 0.0}, 'sagittal': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 1.5873014576404794, 23: 1.8399056597678616, 24: 6.853719017699495, 25: 12.65582199079898, 26: 11.39781048641584, 27: 10.407550849079986, 28: 13.177861952458256, 29: 24.012525279323512, 30: 21.25853430945727, 31: 26.14293596725341, 32: 21.815350038452937, 33: 21.98854117119064, 34: 18.69511018960665, 35: 16.901958133248684, 36: 13.210511188089288, 37: 11.300868899957834, 38: 7.310235368104645, 39: 6.351329708137232, 40: 4.184636509183111, 41: 3.6505068087013655, 42: 3.090849359805651, 43: 2.7797451526868553, 44: 2.5131006996559093, 45: 2.365584231639415, 46: 2.1998977834786335, 47: 2.032113508087695, 48: 1.9180204689486542, 49: 1.8303188822057894, 50: 1.6845210808573854, 51: 1.5904593967727325, 52: 1.4792335381938015, 53: 1.3899589711480171, 54: 1.3218044850942263, 55: 1.2591674887068773, 56: 1.1853684098195005, 57: 1.1379960081957887, 58: 1.0882616735692587, 59: 1.0615451136427667, 60: 1.045303566358712, 61: 1.0187965789804456, 62: 0.9961628262844632, 63: 0.9638562259426778, 64: 0.9557831960488332, 65: 0.9478242428363665, 66: 0.9458730449253071, 67: 0.9450336581541016, 68: 0.942854845244233, 69: 0.941804713091871, 70: 0.9494566518465365, 71: 0.9500606586778445, 72: 0.9642182146709448, 73: 0.9650157113556458, 74: 0.9455894615550086, 75: 0.9308626878142049, 76: 0.9498337572144382, 77: 0.921773197830667, 78: 0.8962286181026204, 79: 0.8705101387839381, 80: 0.8688017423408991, 81: 0.8659260894919805, 82: 0.8685895454248668, 83: 0.863704881938774, 84: 0.8632627161640617, 85: 0.8853587616157026, 86: 0.9295136206288602, 87: 0.9657577779598441, 88: 0.9977236043263996, 89: 1.0300783942610474, 90: 1.0834337508417284, 91: 1.152945411586684, 92: 1.2605524271444748, 93: 1.3354741666938486, 94: 1.3760412864962408, 95: 1.4378795189254143, 96: 1.466284192531992, 97: 1.4327263595476334, 98: 1.3564416528187249, 99: 1.2855946680810193, 100: 1.1973581588485591, 101: 1.1006158638516195, 102: 1.0011546793554418, 103: 0.9369880030921165, 104: 0.906257925676331, 105: 0.8795284939946677, 106: 0.8868466189710872, 107: 0.9001562767542921, 108: 0.9061492484870046, 109: 0.9320318950502321, 110: 0.9404199754144765, 111: 0.9540243780400423, 112: 0.9774052001608038, 113: 0.9759464708391721, 114: 0.983323297333826, 115: 0.9975643206265745, 116: 0.9930844556001929, 117: 1.0076001887860953, 118: 1.024762404727116, 119: 1.0301843457927409, 120: 1.0267746848973147, 121: 1.0285355997076548, 122: 1.0214273109840954, 123: 1.0178275748055823, 124: 1.0079564891749762, 125: 1.010935146909102, 126: 1.0170881099229696, 127: 1.0194764011191209, 128: 1.0425760369514019, 129: 1.0613875056845843, 130: 1.0775408455152258, 131: 1.0971635164429971, 132: 1.1272431980495534, 133: 1.1711917133318508, 134: 1.2174092097154383, 135: 1.2750918498619264, 136: 1.329116634771339, 137: 1.3832116061755588, 138: 1.4498178499681342, 139: 1.5340399235368758, 140: 1.602224383260096, 141: 1.68171383551033, 142: 1.7910265879935654, 143: 1.9063852470684952, 144: 2.029102303406159, 145: 2.1286479491662518, 146: 2.316174411497807, 147: 2.5437759517527794, 148: 2.861685740460422, 149: 3.7585820423706884, 150: 3.240045323001505, 151: 4.187523396667871, 152: 4.818062406928272, 153: 7.449641499298881, 154: 9.625605837773765, 155: 13.217511078136, 156: 18.176028012392024, 157: 21.659211455767917, 158: 21.80050037579349, 159: 18.09608986642609, 160: 14.803253713097126, 161: 16.57239329841778, 162: 14.937247287626615, 163: 17.14611452034945, 164: 10.012760895513585, 165: 9.139676119660923, 166: 6.454936407855863, 167: 4.269540955981448, 168: 2.1058200462029824, 169: 0.2452830185119259, 170: 0.6170212747203001, 171: 0.9999999926470589, 172: 0.0, 173: 0.0, 174: 0.0, 175: 0.0, 176: 0.0, 177: 0.0, 178: 0.0, 179: 0.0, 180: 0.0, 181: 0.0, 182: 0.0, 183: 0.0, 184: 0.0, 185: 0.0, 186: 0.0, 187: 0.0, 188: 0.0, 189: 0.0, 190: 0.0, 191: 0.0}, 'coronal': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0, 30: 0.0, 31: 0.0, 32: 0.0, 33: 0.0, 34: 0.0, 35: 0.0, 36: 0.0, 37: 0.0, 38: 0.4905702960296974, 39: 0.27361784624899665, 40: 3.0856197018737985, 41: 2.020841839250835, 42: 1.5775766749628333, 43: 1.5419926815312381, 44: 2.2073857172264058, 45: 1.8727622546587066, 46: 2.5861632827584113, 47: 2.6244252558345353, 48: 2.822648840723149, 49: 4.10017511818589, 50: 4.726685155066443, 51: 3.9376493308944305, 52: 5.666917267122342, 53: 12.221849468079562, 54: 12.279994280813217, 55: 8.04154912931263, 56: 11.206104458301848, 57: 9.00145648284013, 58: 8.806755338813778, 59: 6.263807431870352, 60: 5.273524475629143, 61: 9.99644189679002, 62: 7.375340505263456, 63: 6.8155407088598485, 64: 8.324619789510601, 65: 5.676832469245084, 66: 8.296021161908849, 67: 5.472052911883454, 68: 5.594015181145007, 69: 4.438853574754369, 70: 3.689999773245738, 71: 4.909084219341716, 72: 5.292105122494933, 73: 4.260957475148364, 74: 5.304093157111648, 75: 3.4218551943054236, 76: 2.8838680400043875, 77: 2.7173022452103766, 78: 2.455458473210996, 79: 2.2459875079232425, 80: 2.144668844921195, 81: 2.9953602607904224, 82: 1.8760712305918472, 83: 1.8279058572953342, 84: 1.7463257354285853, 85: 2.436735993384942, 86: 1.8198230497909749, 87: 1.7250948397747927, 88: 1.7052480660389722, 89: 1.647898351857696, 90: 1.5965029496192111, 91: 1.6061519039931533, 92: 1.586165686696402, 93: 1.6102963974267541, 94: 1.5478416809370446, 95: 1.6003025440200802, 96: 1.6573522499409028, 97: 1.6933156644604828, 98: 1.6692120992450188, 99: 1.7338884277873863, 100: 1.7236904362856227, 101: 1.758064584213522, 102: 1.7577531528730215, 103: 1.67219525240289, 104: 1.7245619893629858, 105: 1.727056702384746, 106: 1.6948033411721362, 107: 1.6882297935854413, 108: 1.6958366228039732, 109: 1.6530399456920022, 110: 1.6529623649766123, 111: 1.6631348125200613, 112: 1.6099767476801508, 113: 1.597112265201956, 114: 1.5931916270657047, 115: 1.5636097142425434, 116: 1.4961086025810677, 117: 1.4791429688480497, 118: 1.4769699326499066, 119: 1.4744385026006945, 120: 1.519984730238607, 121: 1.484962847563239, 122: 1.441260302534945, 123: 1.4577176448301332, 124: 1.4076380660980146, 125: 1.407691378880895, 126: 1.3822692859705876, 127: 1.371532912256348, 128: 1.3705652803487136, 129: 1.3385550119670553, 130: 1.2981653248779268, 131: 1.3027595742977787, 132: 1.2604043272979772, 133: 1.2151625142396942, 134: 1.1950730571453014, 135: 1.1859077625537164, 136: 1.1742777232811221, 137: 1.1545586237907899, 138: 1.178659337702353, 139: 1.1750890127170344, 140: 1.1729951157545275, 141: 1.1745583728987472, 142: 1.1899279177717998, 143: 1.2011002120652505, 144: 1.2061113211141858, 145: 1.232125955899967, 146: 1.2159844632083983, 147: 1.2434799818396578, 148: 1.2678385385421884, 149: 1.275528630266162, 150: 1.2718055944295175, 151: 1.2580425165074627, 152: 1.2957563139388673, 153: 1.2813830285567935, 154: 1.2601468009529384, 155: 1.2583262519166354, 156: 1.262327502344669, 157: 1.233194091562084, 158: 1.1820411122549639, 159: 1.1680555188702688, 160: 1.1572816097191283, 161: 1.1237254894363824, 162: 1.090771056351287, 163: 1.0699557972363785, 164: 1.0385565776928112, 165: 1.0317502076991216, 166: 1.0578707696400684, 167: 1.0249317934834523, 168: 1.0238742743641658, 169: 1.0078249459054742, 170: 0.9913094776631522, 171: 0.9732064259223108, 172: 0.9455291952697062, 173: 0.9499374617725088, 174: 0.947121355619425, 175: 0.9546714288726618, 176: 0.9456257798941786, 177: 0.9499286734105544, 178: 0.9540556822937974, 179: 0.9659055419913757, 180: 0.9877372919975356, 181: 0.9976921142593478, 182: 1.0189574714371534, 183: 1.0348985843321201, 184: 1.0496672947467158, 185: 1.0632205090610811, 186: 1.102379974578872, 187: 1.1398824952851958, 188: 1.1781549313539865, 189: 1.229441012176167, 190: 1.2877184981832226, 191: 1.3139117737089319, 192: 1.3970611793082028, 193: 1.4788135381666967, 194: 1.5482278487299923, 195: 1.6200064587798086, 196: 1.688387590903506, 197: 1.7619356451224757, 198: 1.8757939343938272, 199: 1.9605926973777228, 200: 2.0859239738542956, 201: 2.19745895447415, 202: 2.33187816951005, 203: 2.444256772610069, 204: 2.5530446271154084, 205: 2.7451604110087424, 206: 3.071123196155302, 207: 4.209429671532551, 208: 3.908832268290981, 209: 3.4006634986920896, 210: 3.457904139973815, 211: 4.995156302613588, 212: 5.899855551083984, 213: 4.987002179430611, 214: 4.5801913074976, 215: 5.9254582893242755, 216: 9.790443050873876, 217: 9.076628406679532, 218: 9.044527068618475, 219: 9.618204571682835, 220: 9.747651459496572, 221: 12.285732645615061, 222: 8.907539766701616, 223: 9.10826148972504, 224: 15.263442609304422, 225: 9.75599524478012, 226: 8.350311892231504, 227: 9.707730562525079, 228: 9.606929203442053, 229: 6.831248709361864, 230: 6.331067006907926, 231: 6.034347553284707, 232: 5.681533529288769, 233: 6.084714701986045, 234: 7.550380058783269, 235: 2.6934230793305654, 236: 1.0023719971574887, 237: 0.5250323471888372, 238: 0.20746887923761648, 239: 0.4647887258480461, 240: 0.0, 241: 0.0, 242: 0.0, 243: 0.0, 244: 0.0, 245: 0.0, 246: 0.0, 247: 0.0, 248: 0.0, 249: 0.0, 250: 0.0, 251: 0.0, 252: 0.0, 253: 0.0, 254: 0.0, 255: 0.0}}
		self.slices_to_weight = {'axial':[],'sagittal':[],'coronal':[]}
		self.dice_per_slice = {'axial': {0.0: 62.43631172945505, 1.0: 31.910798121264666, 2.0: 12.419095195145552, 3.0: 9.74543018323048, 4.0: 8.094549590435706, 5.0: 5.245107884401983, 6.0: 4.6310448054672575, 7.0: 3.6775982895590857, 8.0: 2.969582057881445, 9.0: 2.7669152455635686, 10.0: 2.555902769185593, 11.0: 2.0201751895901827, 12.0: 2.131436968159684, 13.0: 1.7670054158487343, 14.0: 1.8558258748377434, 15.0: 1.6737964340314693, 16.0: 1.5887760719409516, 17.0: 1.5172367970044565, 18.0: 1.4390046871063131, 19.0: 1.341177897798571, 20.0: 1.3118912239355895, 21.0: 1.2425237252677674, 22.0: 1.248763173214836, 23.0: 1.1505131899615941, 24.0: 1.2551596701922176, 25.0: 1.0098865591643365, 26.0: 1.0752163453017471, 27.0: 1.100841057441634, 28.0: 1.0040127780062265, 29.0: 1.0582146973792348, 30.0: 0.9206122769273778, 31.0: 0.9852342379995277, 32.0: 0.9712741139332549, 33.0: 0.8771370415153511, 34.0: 0.9749645251348334, 35.0: 0.9788402357897933, 36.0: 0.9523279688913735, 37.0: 0.8970191553706899, 38.0: 0.9992948406143539, 39.0: 0.9230343155693459, 40.0: 1.0164034491433336, 41.0: 0.997011987211474, 42.0: 0.9282106187804857, 43.0: 0.9960405868102971, 44.0: 0.9604464813520986, 45.0: 1.0221188273977517, 46.0: 0.9936522486851793, 47.0: 1.0688377319437832, 48.0: 0.9954919540715574, 49.0: 1.1891159688320645, 50.0: 1.190811514198488, 51.0: 1.2869467900677216, 52.0: 1.187861292925489, 53.0: 1.3620039783575728, 54.0: 1.4102521795002176, 55.0: 1.5658043594001612, 56.0: 1.5345967411474566, 57.0: 1.687291973705332, 58.0: 1.5965385973395811, 59.0: 1.7625485126597051, 60.0: 1.9113022931244528, 61.0: 1.8189780471469479, 62.0: 1.9812795435485708, 63.0: 1.8819660035560455, 64.0: 2.053279550193464, 65.0: 2.36084906926994, 66.0: 2.5841870428670233, 67.0: 2.4769184024170054, 68.0: 3.177947049038081, 69.0: 3.426349151227504, 70.0: 3.4800458301158463, 71.0: 4.079617274988506, 72.0: 3.820178505505333, 73.0: 3.9351808205423797, 74.0: 3.5680893908817577, 75.0: 3.3680899956220705, 76.0: 3.8015203350104207, 77.0: 3.161666741499085, 78.0: 3.1174772011458804, 79.0: 2.9770167290800877, 80.0: 2.9644832824193115, 81.0: 2.7223423977557832, 82.0: 2.706197883724279, 83.0: 2.6593715700449962, 84.0: 2.5457655666286794, 85.0: 2.7236620401382514, 86.0: 2.9675886691327724, 87.0: 3.0197590578993556, 88.0: 3.383428489352464, 89.0: 3.7498952533683267, 90.0: 5.181059749261619, 91.0: 6.118288812889095, 92.0: 7.967679692316831, 93.0: 10.319781971303831, 94.0: 11.84427804124913, 95.0: 11.200985955697679, 96.0: 14.871036417367113, 97.0: 17.154190772224354, 98.0: 20.968355911268514, 99.0: 77.90056060222261, 100.0: 84.90501879980404}, 'sagittal': {0.0: 50.96776705870439, 1.0: 72.00059581054012, 2.0: 32.279009289311766, 3.0: 11.822473839020532, 4.0: 9.555365301236053, 5.0: 6.611923459023167, 6.0: 4.9027687986056865, 7.0: 3.770311565177729, 8.0: 3.0079263040418174, 9.0: 2.8275610393369504, 10.0: 2.820884425938578, 11.0: 2.5789052265230756, 12.0: 2.506112954609077, 13.0: 2.4222388095120095, 14.0: 2.0373581812216304, 15.0: 2.099585942171908, 16.0: 1.9408781182767645, 17.0: 1.7581372125105224, 18.0: 1.533949229209071, 19.0: 1.541335092979716, 20.0: 1.5673493111033454, 21.0: 1.2913535333578217, 22.0: 1.3036564634349999, 23.0: 1.209013681437078, 24.0: 1.2156354494093085, 25.0: 1.08590383026102, 26.0: 1.0860268839307086, 27.0: 1.1380668480608123, 28.0: 1.0990411839875758, 29.0: 1.0258470409629097, 30.0: 1.0851608612570622, 31.0: 1.1391777463461783, 32.0: 0.9422568521113706, 33.0: 1.051599996037345, 34.0: 1.1420012199935272, 35.0: 1.0628224015360606, 36.0: 1.0190244506084591, 37.0: 0.9703743511311845, 38.0: 1.0108452167398123, 39.0: 0.9890013245526049, 40.0: 1.020198364590072, 41.0: 0.9828433444658239, 42.0: 0.9492684432319883, 43.0: 1.1132738342685733, 44.0: 1.0998997609174466, 45.0: 1.1327682763187348, 46.0: 1.2448434849453025, 47.0: 1.3876580415828252, 48.0: 1.5870514792542463, 49.0: 1.4688628768305372, 50.0: 2.028854395969013, 51.0: 1.4825723955110033, 52.0: 1.6698446283054427, 53.0: 1.4300855477637358, 54.0: 1.2540365068176706, 55.0: 1.1131091573061496, 56.0: 1.0524475511463764, 57.0: 1.0797998991617277, 58.0: 0.9186761520882394, 59.0: 1.0042058951106516, 60.0: 1.0934878758616944, 61.0: 1.0548434146184547, 62.0: 1.0593383109716035, 63.0: 1.014999791147995, 64.0: 1.0674339616931245, 65.0: 1.1361934859917686, 66.0: 1.1991444189056557, 67.0: 1.092495646133145, 68.0: 0.9963965061828698, 69.0: 1.1961341236676861, 70.0: 1.1249412612445462, 71.0: 1.0497507165216913, 72.0: 1.1435998766147752, 73.0: 1.186986341944429, 74.0: 1.1340703696993888, 75.0: 1.1271924817291374, 76.0: 1.2510378712307357, 77.0: 1.2599163005973164, 78.0: 1.3147965944208082, 79.0: 1.2991227708366513, 80.0: 1.5851717108550973, 81.0: 1.5622864714886746, 82.0: 1.5364980524846192, 83.0: 1.7630557510613891, 84.0: 1.9152469316985492, 85.0: 2.0174693269071486, 86.0: 2.024638463972173, 87.0: 2.4130894648556818, 88.0: 2.473192656547984, 89.0: 2.413535657438708, 90.0: 2.714598143217218, 91.0: 2.6488140829821814, 92.0: 2.912281195666645, 93.0: 3.5404771348607462, 94.0: 4.431127045887054, 95.0: 5.256978611486352, 96.0: 7.891917269576563, 97.0: 9.348411555848898, 98.0: 27.65505019184757, 99.0: 74.67404953495927, 100.0: 42.679460170111845}, 'coronal': {0.0: 74.10495461701306, 1.0: 48.5451356006471, 2.0: 19.284963595732023, 3.0: 8.98149515024027, 4.0: 8.431401702813297, 5.0: 5.739205620853656, 6.0: 4.624386565372095, 7.0: 4.051877467833782, 8.0: 3.767764199884313, 9.0: 3.3021568175460656, 10.0: 3.184765853724707, 11.0: 2.816240323315641, 12.0: 3.036459733470033, 13.0: 2.455762162077782, 14.0: 2.5814639346505857, 15.0: 2.591212253203977, 16.0: 2.268648827821588, 17.0: 2.2713172984389542, 18.0: 2.045084121606346, 19.0: 2.107170040021537, 20.0: 2.129695843918555, 21.0: 2.0050391424656593, 22.0: 2.1930600928132797, 23.0: 2.238547426505257, 24.0: 2.2435978932909206, 25.0: 2.539061762603491, 26.0: 2.498567142133566, 27.0: 2.6098049430979375, 28.0: 2.70555919773782, 29.0: 2.2425609180173254, 30.0: 2.517840145702801, 31.0: 2.358025658411446, 32.0: 2.06510275194616, 33.0: 2.166738294874575, 34.0: 2.035855008089049, 35.0: 2.136614064995668, 36.0: 2.059849532495768, 37.0: 1.9591496134545574, 38.0: 2.1297826902244275, 39.0: 1.8452248381755445, 40.0: 2.1301070673811724, 41.0: 1.9193418902829604, 42.0: 1.910431119332837, 43.0: 1.86041002249202, 44.0: 1.924008083486842, 45.0: 1.730514218218807, 46.0: 1.8116935761470383, 47.0: 1.6647375761678178, 48.0: 1.610158763816588, 49.0: 1.7248737500305364, 50.0: 1.39947714500157, 51.0: 1.7531314360413819, 52.0: 1.7287508923682302, 53.0: 1.732617943619183, 54.0: 1.8179740020118753, 55.0: 1.7760163302426673, 56.0: 1.9186070105501665, 57.0: 1.8391508467368343, 58.0: 1.722219488629288, 59.0: 1.6890392407948833, 60.0: 1.7233223822752153, 61.0: 1.410773938415908, 62.0: 1.5471676367260558, 63.0: 1.3646936113242218, 64.0: 1.4006372368062925, 65.0: 1.4007472905504614, 66.0: 1.3030414612681454, 67.0: 1.3134519951440584, 68.0: 1.1954041435773455, 69.0: 1.318260186307913, 70.0: 1.3769926238893433, 71.0: 1.194289308582934, 72.0: 1.4300276236858425, 73.0: 1.3692154514065935, 74.0: 1.3766943251650963, 75.0: 1.461130840691347, 76.0: 1.4420980823594727, 77.0: 1.5328369891487226, 78.0: 1.6519136849830274, 79.0: 1.606537440681613, 80.0: 1.8144939762414005, 81.0: 1.8408399674458635, 82.0: 1.9264291776372144, 83.0: 2.2054746835252823, 84.0: 2.351187372149486, 85.0: 2.7532782341300974, 86.0: 2.82963694507355, 87.0: 3.0107632627798164, 88.0: 3.6499437851403878, 89.0: 3.468663916958593, 90.0: 3.7720175871106947, 91.0: 3.8979530419869084, 92.0: 4.431614684563817, 93.0: 4.8767810287875255, 94.0: 5.302216219172037, 95.0: 6.340619626859468, 96.0: 8.26602359072199, 97.0: 8.155689198507806, 98.0: 15.472372620632449, 99.0: 41.365500298650886, 100.0: 74.74369133287416}}

		self.dice_per_slice = {'axial': {0.0: 39.05713662615034, 1.0: 22.231226602311075, 2.0: 15.766805265888411, 3.0: 7.892679119567333, 4.0: 7.3961767955511775, 5.0: 5.664702315937509, 6.0: 4.308532030621958, 7.0: 3.725638474819112, 8.0: 3.2652741650049695, 9.0: 2.893017385942509, 10.0: 2.587577501228712, 11.0: 2.3008421465909037, 12.0: 2.168449844963141, 13.0: 1.907998768500685, 14.0: 1.8919563375250306, 15.0: 1.8343539219661917, 16.0: 1.7410750096527376, 17.0: 1.6111193045855479, 18.0: 1.5355117187523541, 19.0: 1.4397107986290694, 20.0: 1.5459321299169124, 21.0: 1.3148555262567019, 22.0: 1.2735696384303314, 23.0: 1.3295599323717728, 24.0: 1.2693522028145576, 25.0: 1.0875787183730234, 26.0: 1.1783353208171863, 27.0: 1.2155642011222767, 28.0: 1.059362887463191, 29.0: 1.0787046369127906, 30.0: 1.0873140747793992, 31.0: 1.0641865750305808, 32.0: 0.9809409960624241, 33.0: 0.9598474306370398, 34.0: 1.059020227393293, 35.0: 0.9950304242715945, 36.0: 0.9958901360481986, 37.0: 0.9546556335870473, 38.0: 1.0356336734366705, 39.0: 0.9954730240095252, 40.0: 1.0153109987464037, 41.0: 1.0155831336360217, 42.0: 0.9284099171781607, 43.0: 1.010464482353115, 44.0: 1.0216897005783179, 45.0: 0.9776311950717549, 46.0: 1.0390396447225514, 47.0: 1.0250162423948934, 48.0: 1.0080176288168377, 49.0: 1.005733512430719, 50.0: 1.2558981120537136, 51.0: 1.0673676225155684, 52.0: 1.1232666811418603, 53.0: 1.2411227861721372, 54.0: 1.32792079554951, 55.0: 1.3242750044565557, 56.0: 1.3688756306772456, 57.0: 1.5247514709137162, 58.0: 1.3976475253404395, 59.0: 1.6849737756529333, 60.0: 1.7456480292608898, 61.0: 1.7671421555954963, 62.0: 1.8919340160701648, 63.0: 1.8213266211758734, 64.0: 2.0919911755883462, 65.0: 2.0169905056665995, 66.0: 2.2428443931651874, 67.0: 2.1485729954802757, 68.0: 2.375439605695202, 69.0: 2.793246664682175, 70.0: 3.1593688096957426, 71.0: 3.311728414995293, 72.0: 3.4723565020759173, 73.0: 4.201855778577916, 74.0: 4.091031391279178, 75.0: 3.5651938832467778, 76.0: 4.0469370982303055, 77.0: 3.9137725158837466, 78.0: 3.489656021119113, 79.0: 3.409266447215581, 80.0: 3.5332333775500593, 81.0: 3.008205739048379, 82.0: 2.9935854162637834, 83.0: 2.829665714826913, 84.0: 2.7589791861463393, 85.0: 2.719714563903471, 86.0: 2.581198510071079, 87.0: 2.4390639106818144, 88.0: 2.5920375242001836, 89.0: 2.6448919092359304, 90.0: 2.8285668903740766, 91.0: 3.0086951292629625, 92.0: 3.255198047770726, 93.0: 3.2864826226394177, 94.0: 3.8595198069763255, 95.0: 5.056930920719724, 96.0: 6.9990841601046645, 97.0: 7.311956203433219, 98.0: 16.080223556838504, 99.0: 32.19731772539764, 100.0: 28.499534065015776}, 'sagittal': {0.0: 89.27547303163747, 1.0: 52.048302235658234, 2.0: 27.204604181794284, 3.0: 9.456350711264838, 4.0: 8.511008773497958, 5.0: 5.971980689873606, 6.0: 4.285788547773262, 7.0: 3.5852267938657363, 8.0: 2.9242526330730274, 9.0: 2.7869153283029267, 10.0: 2.7120396870055115, 11.0: 2.5902481941841717, 12.0: 2.578421174918688, 13.0: 2.425679792260925, 14.0: 2.0165654207884987, 15.0: 2.032092604673811, 16.0: 1.9136063275935347, 17.0: 1.7458115747190872, 18.0: 1.5753061626625604, 19.0: 1.4399253232174885, 20.0: 1.542173540737191, 21.0: 1.335621313683411, 22.0: 1.3018878812504857, 23.0: 1.216924443866477, 24.0: 1.1830439498323368, 25.0: 1.1446304768616162, 26.0: 1.1635476698413474, 27.0: 1.1819165112145686, 28.0: 1.1783870603819233, 29.0: 1.0871222100769553, 30.0: 1.2133784318136334, 31.0: 1.2016239341330879, 32.0: 1.067658102645696, 33.0: 1.2479945951028093, 34.0: 1.2649832317365217, 35.0: 1.2032636216590138, 36.0: 1.1884937338876798, 37.0: 1.1583143899798323, 38.0: 1.1589597659665167, 39.0: 1.1128328892690575, 40.0: 1.1773331930635358, 41.0: 1.0871997750732616, 42.0: 1.1197083099209633, 43.0: 1.197681039652567, 44.0: 1.2053016133242687, 45.0: 1.4193028480303678, 46.0: 1.4445742373947623, 47.0: 1.4935364746221855, 48.0: 1.8863990514517803, 49.0: 1.6806199475349115, 50.0: 2.2528130592603084, 51.0: 1.6883081177887793, 52.0: 1.9163248706809963, 53.0: 1.505824388570507, 54.0: 1.4505868342929003, 55.0: 1.4120853024504059, 56.0: 1.2063014643796943, 57.0: 1.2013532069161426, 58.0: 1.112560886251048, 59.0: 1.1312015551452919, 60.0: 1.294668611930034, 61.0: 1.1861289399535897, 62.0: 1.2338392499001716, 63.0: 1.2195117564927203, 64.0: 1.210174770850059, 65.0: 1.2542155356695146, 66.0: 1.3280084360430542, 67.0: 1.2969399555700956, 68.0: 1.1196946266663739, 69.0: 1.2829079053785581, 70.0: 1.2877370640402495, 71.0: 1.1423581774543727, 72.0: 1.238947810647386, 73.0: 1.2444991881755854, 74.0: 1.2152411979897226, 75.0: 1.2178076029691915, 76.0: 1.2240265364158642, 77.0: 1.2481819522903086, 78.0: 1.3026031878157163, 79.0: 1.2870670447894994, 80.0: 1.496681293267506, 81.0: 1.412082852879033, 82.0: 1.5384432907208159, 83.0: 1.6694100865854546, 84.0: 1.8102365365611663, 85.0: 1.8999200083457346, 86.0: 1.963207310013686, 87.0: 2.4128266955596676, 88.0: 2.4931629799168684, 89.0: 2.517390205668898, 90.0: 2.6333342868583864, 91.0: 2.6485452046376254, 92.0: 2.7696153744770444, 93.0: 3.306127168226198, 94.0: 3.8787016863918256, 95.0: 4.935969971325316, 96.0: 6.810277383019013, 97.0: 7.223104534654016, 98.0: 20.77273046401085, 99.0: 45.13190067093991, 100.0: 87.88926076645242}, 'coronal': {0.0: 44.03778658890195, 1.0: 35.25749850677885, 2.0: 15.879063919129493, 3.0: 8.209423266177648, 4.0: 7.780564110013715, 5.0: 5.567610874379891, 6.0: 4.422367549157516, 7.0: 4.0988900389464895, 8.0: 3.76255136482649, 9.0: 3.474569750838029, 10.0: 3.2927773303348364, 11.0: 3.064174414835636, 12.0: 3.0775047450572246, 13.0: 2.581584513480213, 14.0: 2.694701419562385, 15.0: 2.6206994507810286, 16.0: 2.3968562340286033, 17.0: 2.3590500225554893, 18.0: 2.2070755490340446, 19.0: 2.214457856042495, 20.0: 2.30913889882242, 21.0: 2.156358134409931, 22.0: 2.30341886900672, 23.0: 2.3588021104528716, 24.0: 2.4330406062498806, 25.0: 2.612403403574755, 26.0: 2.769067899099682, 27.0: 2.7276106815551255, 28.0: 2.6354397054562457, 29.0: 2.533831360224537, 30.0: 2.6151902119477826, 31.0: 2.3645314604478624, 32.0: 2.3510291451886878, 33.0: 2.24625822448316, 34.0: 2.4533558197869083, 35.0: 2.5937060621307317, 36.0: 2.5656076327369552, 37.0: 2.5261807364329645, 38.0: 2.6166619705551026, 39.0: 2.4915853573110422, 40.0: 2.5734463918635866, 41.0: 2.1996303584365666, 42.0: 2.292483960743659, 43.0: 2.3238659864144613, 44.0: 2.111405219299323, 45.0: 2.0538615285375763, 46.0: 1.9450541671589163, 47.0: 1.7167890607656422, 48.0: 1.8083836062233294, 49.0: 1.7839750688257108, 50.0: 1.4723493636008604, 51.0: 1.8156852454515375, 52.0: 1.932773925176436, 53.0: 1.786932324502824, 54.0: 1.9000866312379494, 55.0: 1.9443123945024356, 56.0: 1.8930081139388277, 57.0: 2.0960184162704696, 58.0: 1.8320056931144106, 59.0: 1.7891208224942585, 60.0: 1.8946776336111668, 61.0: 1.7046699933994167, 62.0: 1.6876071620581208, 63.0: 1.527547209694632, 64.0: 1.5431781534793296, 65.0: 1.5558397892521987, 66.0: 1.4873486652906958, 67.0: 1.3629575970594219, 68.0: 1.37959342197103, 69.0: 1.3532878542888236, 70.0: 1.4804637558739153, 71.0: 1.3648265170893246, 72.0: 1.3927212396752926, 73.0: 1.4332718511840845, 74.0: 1.4761594554944735, 75.0: 1.5065062202032191, 76.0: 1.517411160373559, 77.0: 1.5618562468834665, 78.0: 1.6454498214706195, 79.0: 1.655553336930768, 80.0: 1.8712267849517186, 81.0: 1.8748600505826483, 82.0: 2.024091363346253, 83.0: 2.330637917262031, 84.0: 2.5768164446361084, 85.0: 2.9773362762103854, 86.0: 3.1058522232332813, 87.0: 3.404801737530049, 88.0: 3.974267670765911, 89.0: 4.051513644776195, 90.0: 4.1161082157596605, 91.0: 4.3167663566987216, 92.0: 4.740507328310419, 93.0: 5.35749769289956, 94.0: 5.664582042162341, 95.0: 6.98358020669144, 96.0: 9.23621420008446, 97.0: 8.896311884514944, 98.0: 15.688863453944158, 99.0: 34.06702994823622, 100.0: 52.70787213781344}}


		#cutoffs : axial = 5, sagittal - 5, coronal = 2

		self.cutoffs = {'axial':5,'sagittal':5,'coronal':2}

	def run_script(self):
		self.graph_results()



	def modify_view(self):

		t1_mri_path = 'sub-A00033747_ses-NFB3_T1w.nii.gz'
		mask_path = 'sub-A00033747_ses-NFB3_T1w_brainmask.nii.gz'

		t1_img = nib.load(t1_mri_path)
		mask_img = nib.load(mask_path)

		# Get the data from the loaded NIfTI files
		t1_data = t1_img.get_fdata()
		mask_data = mask_img.get_fdata()

		# Set every slice of the axial view to be blank (all black pixels)
		# In the axial view, slices are along the third axis, so we set all slices to 0
		#t1_data[:, :, :] = 0  # Set all values in the T1 MRI data to 0 (black)
		for x in range(0,50):
			mask_data[:, x, :][mask_data[:, x, :] == 1] += 1
		print(np.max(mask_data))

		# Save the modified MRI scan and mask as new NIfTI files
		new_t1_img = nib.Nifti1Image(t1_data, t1_img.affine)
		new_mask_img = nib.Nifti1Image(mask_data, mask_img.affine)

		nib.save(new_t1_img, 'blanked_t1_mri.nii')
		nib.save(new_mask_img, 'blanked_mask.nii')

		print("Blanked axial slices saved successfully.")

	def find_intersecting_points(self):
		mask_path = 'sub-A00033747_ses-NFB3_T1w_brainmask.nii.gz'
		mask_img = nib.load(mask_path)
		mask_data = mask_img.get_fdata()

		x_dim, y_dim, z_dim = mask_data.shape

		sum_of_intersections = 0

		for x in range(x_dim):
		    for y in range(y_dim):
		        for z in range(z_dim):
		            intersection_value = mask_data[x, y, z]
		            if intersection_value > 0:
		            	print(intersection_value)
		            
		            sum_of_intersections += intersection_value

		print(f"Total sum of all intersecting points across axial, sagittal, and coronal views: {sum_of_intersections}")

	def graph_results(self):
		for view, slices in self.dice_per_slice.items():
			slice_list = []
			dice_list =[]
			for slice_num, dice_coef in slices.items():
				if dice_coef > self.cutoffs[view]:
					self.slices_to_weight[view].append(slice_num)
				slice_list.append(slice_num)
				dice_list.append(dice_coef)
			plt.figure(figsize=(8, 5))
			plt.plot(slice_list, dice_list, label='dice', marker='o')  

			plt.title(f'{view} dice coef per slice')
			plt.xlabel('slice')
			plt.ylabel('dice')
			plt.legend()

			plt.grid(True)
			plt.show()
		print(self.slices_to_weight)




if __name__ == '__main__':
		AnalyzePerformance().run_script()
