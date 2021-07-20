import tensorflow as tf
import tensorflow.keras as tk
import nthmc, ftr
import sys
sys.path.append('../lib')
import field

conf = nthmc.Conf(nbatch=16, nepoch=1, nstepEpoch=2048, nstepMixing=128, initDt=0.1, stepPerTraj=10)
nthmc.setup(conf)

# using the exact model from t_topoforce_1.py

op0 = (((1,2,-1,-2), (1,-2,-1,2)),
       ((1,1,2,-1,-1,-2), (1,1,-2,-1,-1,2), (1,2,-1,-1,-2,1), (1,-2,-1,-1,2,1)))
# requires different coefficient bounds:
# (1,2,-1,-2,1,-2,-1,2)
# (1,2,-1,-2,1,2,-1,-2)
# (1,-2,-1,2,1,-2,-1,2)
op1 = (((2,-1,-2,1), (2,1,-2,-1)),
       ((2,2,-1,-2,-2,1), (2,2,1,-2,-2,-1), (2,-1,-2,-2,1,2), (2,1,-2,-2,-1,2)))
fixedP = (1,2,-1,-2)
fixedR0 = (2,2,1,-2,-2,-1)
fixedR1 = (1,1,2,-1,-1,-2)
convP0 = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (3,2), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
convP1 = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (2,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
convR = lambda pad: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
), pad)
conv = lambda: ftr.PeriodicConv((
    tk.layers.Conv2D(2, (3,3), activation='gelu', kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
    tk.layers.Conv2D(2, (3,3), activation=None, kernel_initializer=tk.initializers.RandomNormal(), bias_initializer=tk.initializers.RandomNormal()),
))
transform = lambda: ftr.TransformChain([
    ftr.GenericStoutSmear(((0,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op0, [(fixedP, convP0()), (fixedR0, convR((1,2)))], conv()),
    ftr.GenericStoutSmear(((0,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,0),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((0,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
    ftr.GenericStoutSmear(((1,1),(2,2)), op1, [(fixedP, convP1()), (fixedR1, convR((2,1)))], conv()),
])
ftr.checkDep(transform())
action = nthmc.U1d2(beta=5.0, beta0=5.0, size=(16,16), transform=transform())
loss = nthmc.LossFun(action, cCosDiff=0.0, cTopoDiff=1.0, cForce2=1.0, dHmin=0.5, topoFourierN=1)
x0 = action.initState(conf.nbatch)
mcmc = nthmc.Metropolis(conf, nthmc.LeapFrog(conf, action))

# Get the weights from the end of the first epoch with beta=5
# <ssam -n '/^beta: 5$//end epoch/?^weights?;/^[^ ]/-' t_topoforce_1.log|grep -v '^$'
# Edit x/ +[0-9\-]/s/ /, /
# Edit x/[^,]$/a/,/
weights=list(map(lambda x:tf.constant(x,dtype=tf.float64),
 [0.10777749128601967,
 [[[[0.22503679764390566, 0.0012397187545698828]],
  [[0.32348628870257107, -0.17728572267670742]]],
 [[[0.22115357593295262, 0.23973064257906584]],
  [[0.23010040812536078, 0.17885029974291528]]],
 [[[0.29097176976815653, -0.085772376835684241]],
  [[0.27749117939101836, -0.11677014672534393]]]],
 [0.226234266606396, 0.22771307422335313],
 [[[[0.10173740033592889, 0.074560345852424922]],
  [[-0.0046333704888505654, -0.08092290643821741]],
  [[-0.0017987169219324242, 0.11773054973498503]]],
 [[[0.031477931210962262, -0.21850416505331663]],
  [[0.028929500440203221, -0.26111725764486504]],
  [[-0.045938015786738728, -0.066934999488588942]]],
 [[[0.0073821922472247633, 0.058770624744843646]],
  [[0.084740732104112579, -0.0958906863356449]],
  [[0.0388910219848051, 0.010810667644001268]]]],
 [-0.28856739740454307, 0.3085416008570388],
 [[[[0.26570562586387431, 0.27275197444253596],
   [0.19462336153502643, 0.036872872480873391],
   [-0.12941770575834338, -0.23420708086646114],
   [0.058970422259831796, 0.24802858637564471]],
  [[0.16395948417691575, 0.19571550620508218],
   [0.19580052219138619, 0.046239350251182518],
   [-0.19968391830983667, -0.20016904803265448],
   [0.108668486193887, 0.25793018245998783]],
  [[0.15454797819320509, 0.27277516248204459],
   [0.19099323216207353, -0.023227800266396148],
   [-0.16687806350094431, -0.16844910043819772],
   [0.091873928971369784, 0.19671504495819725]]],
 [[[0.14866651261223032, 0.22973399791355134],
   [0.15668052199227475, 0.250675898528665],
   [-0.076321560110556644, -0.21755168138519823],
   [0.21647212349365508, 0.33548719089383228]],
  [[0.11720412799789269, 0.24211032017289869],
   [0.23032710737259171, 0.20818495862447003],
   [-0.16541405974913592, -0.21594038533194146],
   [0.13876198722806452, 0.18226576113492893]],
  [[0.18855168975233955, 0.31119811680270348],
   [0.16084705951757988, 0.28402004942612707],
   [-0.0894114872947375, -0.19027972231597429],
   [0.16707267941611045, 0.20346806912175347]]],
 [[[0.26853492886676694, 0.27307523171707687],
   [0.15466476621658384, 0.10968081655864537],
   [-0.093951634561922237, -0.15784319053490678],
   [0.056536292951117721, 0.15558096085081619]],
  [[0.19524504545978222, 0.35466274354533717],
   [0.1066948628521493, -0.029873342631350126],
   [-0.074819447262167174, -0.24573074476011234],
   [0.19378441034950122, 0.20159357917541273]],
  [[0.29099325237716467, 0.27188552501672858],
   [0.0905688327204224, -0.016161030925490597],
   [-0.18767922087439665, -0.26099223401324834],
   [0.2897476560342766, 0.22266162388240046]]]],
 [0.24797041263543715, 0.33115073224219116],
 [[[[0.32254540162083506, 0.045218340349724835],
   [0.37463119085837115, 0.18070593405977228]],
  [[0.43377379446885023, 0.14003285027208678],
   [0.37179147028315873, 0.15320853872045598]],
  [[0.36358973341804784, 0.017737805888751693],
   [0.37721894195653127, 0.0594341260671101]]],
 [[[0.43959585894097758, 0.14001874624385444],
   [0.35948933200955735, 0.03689999166104807]],
  [[0.32584323509937341, 0.13323405567927074],
   [0.43602433831088583, -0.004636465791650694]],
  [[0.29550868534913477, 0.16683280338511683],
   [0.32416969805975515, 0.063420335246230572]]],
 [[[0.34349619731794612, 0.087772362414111085],
   [0.39497525237981296, 0.034120373051856286]],
  [[0.38514340643553951, 0.067045713614940022],
   [0.3379170107550849, 0.082893601526349781]],
  [[0.33732202554669316, 0.15233453823724366],
   [0.47719582911158348, 0.13621570862237645]]]],
 [0.3106472545790902, 0.061235210450218409],
 [[[[0.15384723883889409, -0.14714467595749958]],
  [[0.077906411315382965, -0.10082907810448964]]],
 [[[0.048066931484403157, -0.091914581725290864]],
  [[0.1892007918755691, 0.0096839748452531814]]],
 [[[0.03800588059570055, -0.1070183177559767]],
  [[0.10426661411984064, -0.079051281683910929]]]],
 [0.22313981452505816, -0.21181780985085563],
 [[[[-0.042259273774565204, -0.10187452626392604]],
  [[0.10078726770011137, -0.081356945776204018]],
  [[-0.0545769185490932, -0.048365106092601952]]],
 [[[0.016445637512030382, 0.084589285400927458]],
  [[-0.022416076633536793, -0.026346710755292144]],
  [[-0.16617189628516665, -0.016919414525517477]]],
 [[[0.028574696170843919, 0.071483309196031436]],
  [[0.0065697475883587124, -0.016277659375361554]],
  [[-0.07641493228988909, 0.038192226153712878]]]],
 [0.19089274307612264, -0.13671033355070844],
 [[[[-0.14941679727675941, -0.1731151927027155],
   [0.23154781710753117, 0.1593806127583009],
   [-0.12409019876490308, -0.11775633487621934],
   [0.053665726898002836, 0.085987356451775132]],
  [[-0.0927051371163076, -0.236875978581598],
   [0.0832462252759685, 0.21701822977280757],
   [-0.042342054810701149, 0.04666540815721959],
   [0.052433881228515111, -0.0462618575865721]],
  [[-0.11954379982502417, -0.19934021351881251],
   [0.13434104552253837, 0.15219839499111876],
   [-0.12827826971536621, -0.10881624022803449],
   [0.10943395200051524, 0.17087689661492]]],
 [[[-0.10268469957387397, -0.23197867007128778],
   [0.24818231953619366, 0.12411614851489612],
   [-0.15287997270649106, -0.060422316773270916],
   [-0.020960904297649269, 0.035056683280239538]],
  [[-0.092654603111503753, 0.0031261482690687864],
   [0.054993898378270692, 0.14794246425087576],
   [-0.12297389251743705, -0.066599071334260837],
   [0.15177419799336969, -0.054991582496842946]],
  [[-0.17712879689780719, -0.20509406082339085],
   [0.22243876222158626, 0.150454829381274],
   [-0.087173657016459238, -0.11206076287928959],
   [0.061275363413540777, 0.0641591637184482]]],
 [[[-0.17827537779557026, -0.15669075355387838],
   [0.19259331925796755, 0.14849898443590373],
   [-0.11881191518785957, -0.015238118830056503],
   [0.0863915913930915, -0.02638350181603568]],
  [[-0.12033270551342182, -0.10332111382365998],
   [0.057179277687984212, 0.13351160768824205],
   [-0.06541450334851219, -0.033629537018580825],
   [-0.011013791655817324, -0.050349404567349744]],
  [[-0.2207995350889749, -0.073216880362533279],
   [0.080086284262822244, 0.11543659549597532],
   [-0.019737808175371071, -0.06178074499402289],
   [0.049275265646436778, -0.078926136364210989]]]],
 [-0.22361534424689045, -0.23607467697518464],
 [[[[-0.48853142979812136, -0.13015446343897069],
   [-0.38044648785086504, -0.11357726837441405]],
  [[-0.3993883641889584, -0.0078755726785322121],
   [-0.41407664268326017, 0.0363481110954476]],
  [[-0.42219705796759049, 0.027196307420254612],
   [-0.41820656600578365, -0.10502159077979753]]],
 [[[-0.42972777075269858, -0.05032817163624443],
   [-0.46448215204322429, -0.054709012488677419]],
  [[-0.45123289170553166, 0.043078098608112216],
   [-0.48544015072870228, -0.015670159634858755]],
  [[-0.57577243480603346, -0.022378917680670008],
   [-0.45733274088095405, -0.039319853799433029]]],
 [[[-0.43121844622522887, 0.024198945126930528],
   [-0.5184424258812198, -0.02632293725622813]],
  [[-0.47774219540865276, 0.018802308021221573],
   [-0.46818176783468335, -0.020990696694879057]],
  [[-0.48128956398374895, 0.0083904404714300346],
   [-0.46371827107985497, -0.10469979975803455]]]],
 [0.43140700609771832, -0.059653313362744355],
 [[[[-0.047188886283113234, 0.082982981367240541]],
  [[0.0058105396646753243, 0.0173632959756739]]],
 [[[0.016386502930923922, 0.089277097898277383]],
  [[-0.050732396901601542, 0.059884515414856128]]],
 [[[-0.063638307480364351, -0.010700714873783629]],
  [[-0.071065480511488879, 0.048766563518198039]]]],
 [-0.12721088272157891, 0.08384337449713819],
 [[[[-0.04480398357697854, 0.033879343325130588]],
  [[-0.0011718835006789194, 0.17616365205165654]],
  [[-0.019199818571240836, 0.26530980925414971]]],
 [[[-0.076013581995168833, 0.017248118032123245]],
  [[-0.014725354736862225, 0.17931416215744334]],
  [[-0.067460562387712128, 0.099021994934747379]]],
 [[[-0.20370295825823592, 0.22107622353057205]],
  [[-0.022936557968566251, 0.23315856858542414]],
  [[-0.19935375410907827, 0.19678959499025314]]]],
 [-0.12450934722984106, 0.177142635039455],
 [[[[-0.09836436908258156, 0.03508277805479721],
   [0.061766139438134367, -0.077372092307092355],
   [-0.072890032247403577, 0.075840952027707575],
   [0.0895522676383329, -0.00860843321593497]],
  [[-0.070614482183341834, 0.158885973754935],
   [0.028855787592801267, -0.035970071952402177],
   [0.025108344110924514, -0.1027623320687024],
   [0.036526126059257036, -0.038725875011579765]],
  [[-0.076604991411948431, 0.10805574750205492],
   [0.07532783514871691, -0.0020707005428460576],
   [-0.075636845127509691, -0.040769818044052358],
   [0.15443501439510796, -0.084042791317617285]]],
 [[[-0.059250576824816083, 0.12845587505020611],
   [0.18607386192056122, 0.025703778216153959],
   [-0.06344190989729992, -0.011580928867867662],
   [0.0956461354737694, 0.0053502909893612654]],
  [[-0.073903973607164053, 0.088568366469713056],
   [0.025590959832610553, 0.016768618143979733],
   [-0.18994812673868308, 0.12248129065886683],
   [0.093267010805056574, 0.0020287551217652981]],
  [[-0.11849432174921268, 0.051971403243898144],
   [0.062578547759423286, -0.097933929984121329],
   [-0.0065334879956641312, 0.035632272105951025],
   [0.055385992550163715, -0.088313894219177916]]],
 [[[-0.14593775105958007, 0.073148599201545456],
   [0.0433237535983725, -0.080310006606640963],
   [-0.075528487225978741, 0.01935614902067611],
   [0.12729262025434318, -0.089000714053383834]],
  [[-0.12736954845666723, -0.015207930242085076],
   [-0.016333125898110652, -0.04369672089800667],
   [-0.10983940840817585, 0.026181998856211256],
   [0.067701947429180245, -0.071355372107141893]],
  [[-0.063650618455316249, 0.0812374157880422],
   [0.087216509080305316, -4.6125047255853036e-05],
   [-0.039625510295682509, 0.0978459172459997],
   [0.25042695131889708, -0.24477215688628851]]]],
 [0.16593672230766879, -0.12612113154836074],
 [[[[0.14075163210279712, -0.010194045474975408],
   [-0.11642207201925846, -0.032460908312200124]],
  [[0.15792693197982979, 0.023067778210716976],
   [-0.17198780445651993, -0.024884248389886444]],
  [[0.012624103574569589, -0.0010681898823150298],
   [-0.14784915348302843, 0.035041148213658023]]],
 [[[0.18722052669993441, -0.014259625333507598],
   [-0.18367728265939942, 0.0829960349334297]],
  [[0.10798356590310212, -0.0143536896451469],
   [-0.16525448701989298, -0.070972226715234121]],
  [[0.17624533598016651, -0.051749725097265616],
   [-0.17034273642471207, 0.011645509800922027]]],
 [[[0.17470136494540114, 0.097814877168639111],
   [-0.14974787739944512, -0.13712672561976658]],
  [[0.096098816455247135, -0.041259333129046012],
   [-0.095740818456362911, 0.015785909745121077]],
  [[0.14701423165783772, 0.085418314091491573],
   [-0.12019665158958433, -0.035874817923420371]]]],
 [0.029154347039883829, -0.038107712733878508],
 [[[[0.33200481722802172, -0.19139249588980392]],
  [[0.3009136190523134, -0.16098479882674405]]],
 [[[0.3031953140603163, -0.21633440357210248]],
  [[0.33093374235898104, -0.14976663231810289]]],
 [[[0.32614590998513654, -0.24410324415115153]],
  [[0.335032942684826, -0.24685184077447939]]]],
 [0.29746878727319659, -0.335425070096746],
 [[[[0.1689647153337345, 0.063064055131676067]],
  [[0.041107189163552667, 0.11599384254673757]],
  [[-0.045866689368077973, 0.078991355524193937]]],
 [[[0.19823346928341209, 0.045257568351238966]],
  [[0.16996676699505739, 0.044046850627246295]],
  [[-0.04363956128949309, 0.16319570815657575]]],
 [[[0.34949705207591242, -0.051229502989261121]],
  [[0.28563673735673989, 0.11243112854017887]],
  [[0.16888630626014109, 0.0078943052702984389]]]],
 [0.19122670453262774, -0.25461884448845912],
 [[[[-0.049787081273891219, 0.25294010389907795],
   [0.1297477520142325, -0.25910436826196204],
   [0.071378812275118539, 0.17203055009431548],
   [0.063274439379184089, -0.31617927449394695]],
  [[-0.08930709870259379, 0.40355442227472116],
   [0.04542207791013967, -0.26364923442531213],
   [0.10053089599881877, 0.14867731211650279],
   [-0.20029962259300454, -0.29684339747711247]],
  [[-0.10750833172674308, 0.4085589481360643],
   [0.1199003422543551, -0.23250162282965767],
   [0.0071483246767227778, 0.017750367617434468],
   [-0.047686819078275725, -0.30480276278914442]]],
 [[[-0.14663793767378361, 0.32868532120054622],
   [0.096775718745396269, -0.36228399098030539],
   [-0.015072852257448796, 0.12745415331870016],
   [0.022293545124960411, -0.27372858936880579]],
  [[-0.093690068499167944, 0.31009004740924145],
   [0.1360826934802245, -0.30282926419600703],
   [0.10541310391840364, 0.020622097641886428],
   [-0.0140198168822084, -0.207053835654591]],
  [[-0.172015999212022, 0.33570835270846094],
   [0.12860000938526833, -0.37312886431357345],
   [0.14695522564649852, 0.11922561212543425],
   [0.05546908743019547, -0.19608398266352384]]],
 [[[-0.074071943834941423, 0.333009195396359],
   [0.00752672403196632, -0.2667027858650815],
   [-0.054098651436622791, 0.31826492290173652],
   [0.11220311607937007, -0.30554297009209608]],
  [[-0.065623148926824665, 0.27911542722100041],
   [0.0837805981769425, -0.36998086796550339],
   [0.10029595721175164, 0.27217849859627752],
   [-0.031112600238331685, -0.30215992812155384]],
  [[-0.16528648398082182, 0.23004134558125569],
   [0.16104332486121559, -0.36406340670587534],
   [0.077780718067558263, 0.13938399705297952],
   [0.072049133182911618, -0.16727821862384012]]]],
 [-0.16835079887229212, 0.26936703202365775],
 [[[[-0.28988154160139251, -0.15985985358934171],
   [0.42454858948088542, 0.26634693035136686]],
  [[-0.30456771493063195, -0.154237827238576],
   [0.43813841065973264, 0.18023070503675229]],
  [[-0.34748806934601834, -0.15175300595027327],
   [0.36246361990165987, 0.15616465770304849]]],
 [[[-0.35462779184811227, -0.20177103900387669],
   [0.44493128522248659, 0.16429002374402502]],
  [[-0.374332057555931, -0.22805330196071918],
   [0.33667700533197081, 0.19843618437329721]],
  [[-0.27874943891937326, -0.16517956591340766],
   [0.38827154909376754, 0.20548095320593526]]],
 [[[-0.28237493249117251, -0.23308798935436961],
   [0.4183813579639018, 0.27663791884932765]],
  [[-0.33756018438735463, -0.22331966261069761],
   [0.502865876110372, 0.24057172416730274]],
  [[-0.394314677086443, -0.15014960394726559],
   [0.42778101380282668, 0.19826844942407851]]]],
 [0.32912653371071338, 0.12971987733692586],
 [[[[-0.23121680141452555, 0.20663577890100804]],
  [[-0.15510414532725758, 0.11544146324125637]],
  [[-0.14678293986165278, 0.17879276854454854]]],
 [[[-0.17577819436193826, 0.11857749387290313]],
  [[-0.096652675252117154, 0.060772954714413]],
  [[-0.14635224424409812, 0.20303788910051779]]]],
 [-0.29191695564301062, 0.22214143336370348],
 [[[[0.063275926413741176, -0.019821660075772886]],
  [[0.11900999908095332, -0.064632979330447052]],
  [[0.07947137134492, -0.011409268588315518]]],
 [[[0.080190411411195137, 0.16363676503627925]],
  [[0.24331421612707452, 0.098347242012079589]],
  [[0.21858535626653966, 0.19898117820463512]]],
 [[[0.13439407897558694, 0.17594334657347707]],
  [[0.184482665600755, 0.086558354833436352]],
  [[0.098895021377218062, 0.10034340749609234]]]],
 [0.28766446922639366, 0.18830283216389931],
 [[[[-0.24014528752774361, -0.1749921567297667],
   [0.252618440350605, 0.1861101081402689],
   [0.13475070999449953, 0.20440388896248077],
   [0.089040080506145619, 0.016091289760381572]],
  [[-0.24919248312743128, -0.34951461685846197],
   [0.21591684006937609, 0.085077487453570938],
   [0.09627687458335081, 0.11911235828883883],
   [0.1561620480068118, 0.0728680650601261]],
  [[-0.3110988253830072, -0.28780467525033143],
   [0.12305080858977076, 0.23404977845557529],
   [0.1598452745113487, 0.22147029365327],
   [0.21147177394640798, 0.091699880089057217]]],
 [[[-0.17678997154715131, -0.28169785585971996],
   [0.20050897957924466, 0.18807946555267127],
   [0.17069797319237831, 0.25079668803469907],
   [0.12458008575369732, 0.21045904747927319]],
  [[-0.21690651735912275, -0.25284488836197788],
   [0.17921960268208639, 0.11512998808788276],
   [0.098868064218252408, 0.19390226549971534],
   [0.085593899789960848, 0.0826918192338936]],
  [[-0.25617252458135237, -0.27706278520449223],
   [0.16896527427180524, 0.18019329917677751],
   [0.095545345902461845, 0.16134561257758628],
   [0.049152743636188219, 0.096805578417092766]]],
 [[[-0.24114459902432267, -0.28423064207332849],
   [0.21033926736626893, 0.19580103687699227],
   [0.25384826167722124, 0.22153933865127856],
   [0.1039025475099581, 0.18108706663999796]],
  [[-0.24422141253716786, -0.24955438795353252],
   [0.24387109733650622, 0.23406951739511098],
   [0.18857424515065843, 0.21258210513461176],
   [0.17905282111768528, 0.1405562598635]],
  [[-0.18991662379385818, -0.34583185108652043],
   [0.15177299999170232, 0.1902513157759996],
   [0.14398896686113274, 0.22203113091323723],
   [0.1712809813093577, 0.095777377349607951]]]],
 [0.266459240622586, 0.22464338377193793],
 [[[[0.54642876120140671, 0.022255414127262712],
   [0.60460345194518761, -0.012547969118230813]],
  [[0.5923629962475554, -0.022920177103002787],
   [0.53774728608155042, 0.0058482756392089896]],
  [[0.65121326507442168, 0.042490718705521471],
   [0.65309514305939642, -0.035627808740578137]]],
 [[[0.523798397900355, -0.058996574033881449],
   [0.5162779820976291, 0.051244359961269553]],
  [[0.49432136279989775, 0.054686862253498253],
   [0.58952384506044164, -0.0016246302307039101]],
  [[0.568024585872597, -0.060047256772231827],
   [0.60574357954364466, -0.0059065325039183852]]],
 [[[0.51260073937381789, -0.026304691253586691],
   [0.49601296398979744, 0.00786045436805897]],
  [[0.53879238103777793, 0.01407545911362273],
   [0.53367507550239968, 0.0074507463079518579]],
  [[0.5341928505990079, -0.097093686786546507],
   [0.52920439004935826, -0.12443258643827179]]]],
 [0.40513499446087387, 0.0018223248002861966],
 [[[[-0.021055791220898727, -0.02586652508390647]],
  [[0.0018249054459674318, 0.038461976504620063]],
  [[-0.020088528834203107, -0.011314540756780914]]],
 [[[-0.0084842092981042491, -0.031201794083018659]],
  [[-0.0052521144458342337, -0.030680503687195656]],
  [[0.012619442316836369, -0.047126984424972075]]]],
 [-0.25816012000905691, -0.18593284588279355],
 [[[[0.29211185245053956, 0.26671194126255349]],
  [[0.3208067301273419, 0.17257489517719163]],
  [[0.34143603336491263, 0.32841331143163921]]],
 [[[0.3799470631104685, 0.36175738183829254]],
  [[0.33523423424505089, 0.26450821548254388]],
  [[0.3569200482079728, 0.38558248315390081]]],
 [[[0.3280725650256115, 0.36737479003087403]],
  [[0.22507580761294169, 0.2405003128858168]],
  [[0.2818894394796651, 0.37185125534419905]]]],
 [0.31374265843804855, 0.20333508765027244],
 [[[[-0.22196068033738076, -0.21503273230376296],
   [-0.13273404463028271, -0.22699930769690196],
   [0.30248183876843004, 0.31051899281129075],
   [0.36383674301651353, 0.23285773299641943]],
  [[-0.24970111686801241, -0.18856415136254792],
   [-0.14368115240114587, -0.19388904519621963],
   [0.39561316325866913, 0.2583939466967779],
   [0.38197656714285244, 0.36216433394645603]],
  [[-0.23125309603948918, -0.23846205380223456],
   [-0.19949086425302204, -0.13961198027055316],
   [0.37433570532023974, 0.26632291005205555],
   [0.289186090563434, 0.30292629948222666]]],
 [[[-0.24173955562723018, -0.33213331316952316],
   [-0.25243546254074817, -0.054434606896342386],
   [0.33969943048853846, 0.33139089127949795],
   [0.3811223211620457, 0.2661423956712381]],
  [[-0.27440835649106543, -0.10482261707086496],
   [-0.17991482015027516, -0.175753893201862],
   [0.38933472922873952, 0.28601196041375132],
   [0.22805217715242551, 0.32885604067667334]],
  [[-0.21798011770561712, -0.18730883081467453],
   [-0.144973789547885, -0.21037697739961822],
   [0.39616877309157705, 0.21563835620504185],
   [0.34753791761630881, 0.34181116092421704]]],
 [[[-0.22202689814075449, -0.29317440709582121],
   [-0.18601975026388956, -0.13675553046274797],
   [0.3277336616416836, 0.29683477896643318],
   [0.35938380172100987, 0.2205854131401396]],
  [[-0.26203211192682324, -0.19868300120207255],
   [-0.18157436354646475, -0.19540683891930558],
   [0.3232796289573005, 0.26765495117326554],
   [0.33851543865741884, 0.26945356952849236]],
  [[-0.1917989993660533, -0.25065821593091714],
   [-0.23658240026535909, -0.25160573835439104],
   [0.27276692585262852, 0.28795966990312055],
   [0.23396555910284497, 0.26102199715667646]]]],
 [0.32920541670370529, 0.28323397673228951],
 [[[[0.5806242152974106, 0.15589731517921418],
   [0.53215104030671667, 0.15578614431372312]],
  [[0.6084766181454333, 0.19120637977293728],
   [0.65318237144365265, 0.22372852051452091]],
  [[0.631614598066587, 0.063713793859909745],
   [0.72692371452393645, 0.1708936537994234]]],
 [[[0.709942474042066, 0.22357420329889327],
   [0.61390957359823717, 0.238081311466899]],
  [[0.67378327590647291, 0.1409156575890024],
   [0.56520374263554374, 0.19687101141470303]],
  [[0.72699169631833593, 0.1421653342299404],
   [0.5939863079804919, 0.18633539626447232]]],
 [[[0.64927934920851327, 0.1488382109347241],
   [0.61794267209026277, 0.18749812861888418]],
  [[0.62007635798447547, 0.06464227460206308],
   [0.617050388006915, 0.19901047350580958]],
  [[0.63656162551304674, 0.14739687992223538],
   [0.534842538691188, 0.13443222755958745]]]],
 [0.44496329774772159, 0.12022454677295409],
 [[[[0.20071860011776305, 0.40915606507053537]],
  [[0.35173293710529685, 0.35976710690605712]],
  [[0.34893398604436066, 0.24645735953062775]]],
 [[[0.29120253853846756, 0.35033877234165212]],
  [[0.34948479262881649, 0.40311631311486185]],
  [[0.24076739653213097, 0.28903070761567051]]]],
 [0.41637018940294623, 0.41815344571576529],
 [[[[-0.10740263478901763, -0.12114134533485318]],
  [[-0.18743479012822825, -0.18578284400190676]],
  [[-0.1113085900349866, -0.175459396450182]]],
 [[[-0.049799687214337425, -0.089579996816453691]],
  [[-0.12117891728796373, -0.19713944632177269]],
  [[-0.10971995710539977, -0.13342102736443623]]],
 [[[-0.092526841504712867, -0.054184106472728123]],
  [[-0.14184165981172175, -0.085363578772297832]],
  [[-0.058545684127059405, -0.057627003523361947]]]],
 [-0.24356080417720438, -0.16473907527383336],
 [[[[0.30647919324573003, 0.28647826628140138],
   [0.27173722255218447, 0.2786496303452542],
   [-0.31392949143179905, -0.36716676713030372],
   [-0.23880940741049059, -0.18061003252415295]],
  [[0.26614829303556464, 0.442730226891593],
   [0.47878438495170228, 0.40431777378853917],
   [-0.3125048058813526, -0.28843922827364571],
   [-0.18714630460909151, -0.08362217812887876]],
  [[0.30425442058712987, 0.31603001464577346],
   [0.363107609941082, 0.35449013420071496],
   [-0.34565173379445729, -0.30546718818303126],
   [-0.30305511170476734, -0.27379661525375637]]],
 [[[0.38433456647662606, 0.22830418366502897],
   [0.31513387911835966, 0.30837711153499514],
   [-0.2480215289130597, -0.26674034234990723],
   [-0.16869261536482344, -0.30427039539152706]],
  [[0.35917097065765591, 0.50087744297984771],
   [0.34552414312348678, 0.39521912617112076],
   [-0.39720244927500903, -0.28529125846904779],
   [-0.28428901477647639, -0.24265021324348729]],
  [[0.32821812954920343, 0.24599615589149559],
   [0.337367900008338, 0.24809204906838841],
   [-0.22998935909055312, -0.31799509088569267],
   [-0.24231071226225598, -0.3203335078916007]]],
 [[[0.35625012001004275, 0.27936178490043845],
   [0.33076213824599771, 0.33118318532232094],
   [-0.26026562980035595, -0.290244262213919],
   [-0.19926110378399448, -0.29737048244889197]],
  [[0.36228259396620571, 0.47197019502301779],
   [0.42602764406857152, 0.40226700611708682],
   [-0.25799199774837706, -0.33525889503588108],
   [-0.23458390093411371, -0.2839486400215977]],
  [[0.43910276555981603, 0.24665257684971592],
   [0.40595355916767273, 0.40134551559417164],
   [-0.31897742622147707, -0.28485422854100295],
   [-0.22878301828359882, -0.24072803271804769]]]],
 [0.32942948886123868, 0.37438573203963177],
 [[[[0.619275757196031, 0.22451259941862209],
   [0.52370394023966027, 0.22204302986865163]],
  [[0.61458809201984144, 0.13637504910885828],
   [0.58808226930739, 0.23916926835609673]],
  [[0.64863908496803924, 0.24269385631485943],
   [0.623220363844448, 0.11525013913977164]]],
 [[[0.62333581559139806, 0.19853559526911413],
   [0.66411777679346129, 0.23501493808610646]],
  [[0.61028050519288868, 0.24991095037841488],
   [0.50629533014839534, 0.18520893719340745]],
  [[0.68735779269359143, 0.25125307043150186],
   [0.69172472456840806, 0.15627959936709043]]],
 [[[0.69213920171935717, 0.16076924275437063],
   [0.68054241897992329, 0.184778090765597]],
  [[0.66995395219053366, 0.23641236574492311],
   [0.46192067077632459, 0.11499131971078018]],
  [[0.635561715456977, 0.15884741268517227],
   [0.66128688246363088, 0.1745734866305404]]]],
 [0.50968210964383831, 0.13483481784897769],
 [[[[0.45544797544727467, 0.46780720870457204]],
  [[0.4515417674796916, 0.42658106186506789]],
  [[0.42871152814332758, 0.456416043964214]]],
 [[[0.48447858531893945, 0.5014023198384624]],
  [[0.51354416440475092, 0.53256793703656147]],
  [[0.42825913369094837, 0.51038933693577337]]]],
 [0.49779957551859461, 0.46865436227163315],
 [[[[0.43297680434967833, -0.058356251595088476]],
  [[0.42920857770875415, -0.10010671281966442]],
  [[0.45395949984913686, -0.077844255997998157]]],
 [[[0.46562143527821015, -0.083290639758746524]],
  [[0.29933290874812779, -0.090637289609323643]],
  [[0.51544623761483355, -0.080506935806990171]]],
 [[[0.4672309159270937, -0.060499224681909304]],
  [[0.39200084793506346, -0.091375742178474773]],
  [[0.41549168028920025, -0.061052530394696623]]]],
 [0.42418169396227412, -0.36794965480257075],
 [[[[0.0020331667000982588, 0.43105863953357343],
   [0.01872877706015566, 0.40360802367719167],
   [-0.032434365301186013, 0.46944606245225051],
   [0.058912641170208585, -0.37976132200129786]],
  [[-0.042987604389272308, 0.50740966440430268],
   [-0.094051634275435614, 0.5040317679759404],
   [0.045219461799633244, 0.49136219560476813],
   [0.036991366404715245, -0.37676975642191973]],
  [[0.012728590145076655, 0.48023300365421068],
   [-0.11062597830784766, 0.42009435036876253],
   [0.053769462175215277, 0.4476564288493331],
   [0.092389215339452838, -0.46734032515564111]]],
 [[[-0.0651606579120671, 0.36580908497315229],
   [-0.026908049444950358, 0.38502517580112089],
   [0.014237198597681312, 0.45311623178602733],
   [-0.017293468936163771, -0.33896427549541014]],
  [[-0.015760722251771418, 0.48349942389658795],
   [-0.089814353374793029, 0.42356951399984022],
   [-0.0027642288358832433, 0.35682687021862997],
   [0.12694835175576591, -0.41883385709793863]],
  [[-0.01505979965443598, 0.45289264164038923],
   [0.049830490290530022, 0.477023802939525],
   [0.0024441492367349069, 0.49234897527303356],
   [0.079836400071219643, -0.39526447142184351]]],
 [[[-0.043182293430475846, 0.44073383616578465],
   [-0.0052617098903903238, 0.43921677151588434],
   [0.030902642343940086, 0.44620171965096778],
   [0.053905337581670425, -0.45225500027669452]],
  [[0.022470299062991288, 0.45643262821437608],
   [0.023539931840548138, 0.45189349120078665],
   [-0.028753202946331533, 0.43160644255169817],
   [0.12078707347496458, -0.3189637840771995]],
  [[0.00070435303761966456, 0.42253725528917724],
   [-0.039558597374626421, 0.41197582057966664],
   [-0.055364945394391234, 0.35999025413545355],
   [0.05671247179843876, -0.39626017321045404]]]],
 [-0.12711454689507479, 0.44371427464912933],
 [[[[-0.52287015008549953, -0.26654278282034771],
   [0.63002725405059, 0.39591526568011481]],
  [[-0.47454111156998252, -0.22949321796725949],
   [0.6125159411307104, 0.34476283606846614]],
  [[-0.50827299688589689, -0.24446505329299],
   [0.60871933707034331, 0.27994683449469893]]],
 [[[-0.55671128416967, -0.339753672558378],
   [0.62855543974911965, 0.29991825788386789]],
  [[-0.51739186894229827, -0.31386534142510086],
   [0.65740896979488284, 0.2980730598250082]],
  [[-0.58955643217662668, -0.25143033549613153],
   [0.66826801611811049, 0.26642244804437931]]],
 [[[-0.48207600331399242, -0.32908440693190705],
   [0.53832755381275077, 0.3736192409338443]],
  [[-0.52171788160362942, -0.23887015050480473],
   [0.613420125109463, 0.28125448322363039]],
  [[-0.46155605044947162, -0.28611075263710462],
   [0.65106774628381825, 0.38642894672414629]]]],
 [0.51007706327196545, 0.29450536216676226]]
))
# The printed weights are from trainable_weights, we need to append beta at the end.
# infer is going to rewrite beta using mcmc's methods.
weights.append(tf.constant(action.beta,dtype=tf.float64))
# reduce the step size (the 0th in weights) for larger beta or more steps/traj
weights[0] = 0.9*weights[0]

nthmc.infer(conf, mcmc, loss, weights, x0, detail=False)
