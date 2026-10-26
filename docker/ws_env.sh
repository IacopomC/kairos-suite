# Sourced INSIDE the Kairos container, before any script that imports the Kairos
# bindings. Sets up the ROS distro the image ships and the built workspace,
# whichever tool built it: colcon (/ws/install) or catkin (/ws/devel).
#
#   source /ws/src/kairos-suite/docker/ws_env.sh
#
# ROS's setup.bash references unbound vars, so nounset is off while sourcing and
# on afterwards, as every caller here runs under `set -u`.
set +u
for _ros_setup in /opt/ros/*/setup.bash; do
  [ -f "$_ros_setup" ] && source "$_ros_setup"
  break
done
if [ -f /ws/install/setup.bash ]; then
  source /ws/install/setup.bash
elif [ -f /ws/devel/setup.bash ]; then
  source /ws/devel/setup.bash
else
  echo "ws_env: no built workspace at /ws/install or /ws/devel" >&2
  return 1
fi
unset _ros_setup
set -u
# A dev build compiles the Kairos bindings into the mounted source tree; a
# self-contained image installs them as a package, which this path would hide.
if compgen -G "/ws/src/kairos/python/src/kairos_python/_kairos_bindings*.so" >/dev/null; then
  export PYTHONPATH=/ws/src/kairos/python/src:${PYTHONPATH:-}
fi
