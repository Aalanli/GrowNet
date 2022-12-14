using GLMakie
GLMakie.activate!()

scene = Scene(backgroundcolor=:white)
cam3d!(scene)

# One can set the camera lookat and eyeposition, by getting the camera controls and using `update_cam!`
camc = cameracontrols(scene)
update_cam!(scene, camc, Vec3f(0, 8, 0), Vec3f(4.0, 0, 0))


t = 0:0.1:15
u = -1:0.1:1
x = [u * sin(t) for t in t, u in u]
y = [u * cos(t) for t in t, u in u]
z = [t / 4 for t in t, u in u]
surface!(scene, x, y, z; colormap = [:orangered, :orangered],
    lightposition = Vec3f(0, 0, 0), ambient = Vec3f(0.65, 0.65, 0.65),
    backlight = 5.0f0, figure = (; resolution = (1200, 800)))
wireframe!(scene, x, y, z, overdraw = true, linewidth = 0.1) # try overdraw = true
