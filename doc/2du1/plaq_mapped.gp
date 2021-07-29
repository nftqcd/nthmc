set grid lc rgb '#f0f0f0'
set xlabel '$\beta$'
set ylabel 'plaquette'
set key left Left reverse
plot [2:8]\
	"<awk '/^beta: /{b=$2} /^plaq: /{print b,$2,$4}' ../../u1_2d/stats/t_force_b25_0_plaq_post_train" u 1:2:3 w e t 'target',\
	"<awk '/^beta: /{b=$2} /^plaqMapped: /{print b,$2,$4}' ../../u1_2d/stats/t_force_b25_0_plaq_post_train" u 1:2:3 w e t 'mapped',\
	besi1(2.5)/besi0(2.5) w l t '$\beta=2.5$ exact'
