/*
 * Copyright (c) 2008, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "dddmr_rviz_default_plugins/tools/select/edge_selection_tool.hpp"
#include <OgreRay.h>
#include <OgreSceneManager.h>
#include <OgreCamera.h>
#include <OgreMovableObject.h>
#include <OgreRectangle2D.h>
#include <OgreSceneNode.h>
#include <OgreViewport.h>
#include <OgreMaterialManager.h>
#include <OgreTexture.h>
#include <OgreTextureManager.h>

#include <QKeyEvent>  // NOLINT cpplint cannot handle include order

#include "dddmr_rviz_default_plugins/tools/move/move_tool.hpp"

#include "rviz_common/interaction/selection_manager.hpp"
#include "rviz_common/display_context.hpp"
#include "rviz_common/render_panel.hpp"
#include "rviz_common/display.hpp"
#include "rviz_common/tool.hpp"
#include "rviz_common/viewport_mouse_event.hpp"
#include "rviz_common/load_resource.hpp"

namespace dddmr_rviz_default_plugins
{
namespace tools
{


EdgeSelectionTool::EdgeSelectionTool()
: Tool(),
  move_tool_(new MoveTool()),
  selecting_(false),
  sel_start_x_(0),
  sel_start_y_(0),
  moving_(false)
{
  shortcut_key_ = 's';
  access_all_keys_ = true;
}

EdgeSelectionTool::~EdgeSelectionTool()
{
  delete move_tool_;
}

void EdgeSelectionTool::onInitialize()
{
  move_tool_->initialize(context_);
  //@ ros stuff can only be placed here instead of in constructor
  rclcpp::Node::SharedPtr raw_node = context_->getRosNodeAbstraction().lock()->get_raw_node();
  selected_edge_pub_ = raw_node->create_publisher<std_msgs::msg::String>("edge_selection_msg", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());
}

void EdgeSelectionTool::activate()
{
  setStatus("Click and drag to select objects on the screen.");
  context_->getSelectionManager()->setTextureSize(512);
  selecting_ = false;
  moving_ = false;
//  context_->getSelectionManager()->enableInteraction(true);
}

void EdgeSelectionTool::deactivate()
{
  context_->getSelectionManager()->removeHighlight();
}

void EdgeSelectionTool::update(float wall_dt, float ros_dt)
{
  (void) wall_dt;
  (void) ros_dt;
  auto sel_manager = context_->getSelectionManager();

  if (!selecting_) {
    sel_manager->removeHighlight();
  }
}

int EdgeSelectionTool::processMouseEvent(rviz_common::ViewportMouseEvent & event)
{
  auto sel_manager = context_->getSelectionManager();

  int flags = 0;

  if (event.alt()) {
    moving_ = true;
    selecting_ = false;
  } else {
    moving_ = false;

    if (event.leftDown()) {
      selecting_ = true;

      sel_start_x_ = event.x;
      sel_start_y_ = event.y;
    }
  }

  if (selecting_) {
    sel_manager->highlight(
      event.panel->getRenderWindow(),
      sel_start_x_,
      sel_start_y_,
      event.x,
      event.y);

    if (event.leftUp()) {
      rviz_common::interaction::SelectionManager::SelectType type =
        rviz_common::interaction::SelectionManager::Replace;

      rviz_common::interaction::M_Picked selection;

      if (event.shift()) {
        type = rviz_common::interaction::SelectionManager::Add;
      } else if (event.control()) {
        type = rviz_common::interaction::SelectionManager::Remove;
      }

      sel_manager->select(
        event.panel->getRenderWindow(),
        sel_start_x_,
        sel_start_y_,
        event.x,
        event.y,
        type);

      selecting_ = false;
      publishSelected(event);
    }

    flags |= Render;
  } else if (moving_) {
    sel_manager->removeHighlight();

    flags = move_tool_->processMouseEvent(event);

    if (event.type == QEvent::MouseButtonRelease) {
      moving_ = false;
    }
  } else {
    sel_manager->highlight(
      event.panel->getRenderWindow(),
      event.x,
      event.y,
      event.x,
      event.y);
  }

  return flags;
}

void EdgeSelectionTool::publishSelected(rviz_common::ViewportMouseEvent & event){
  auto sel_manager = context_->getSelectionManager();
  rviz_common::interaction::M_Picked selection = sel_manager->getSelection();
  rviz_common::properties::PropertyTreeModel* model = sel_manager->getPropertyModel();
  for(auto it=selection.begin(); it!=selection.end();it++){
    
  }
  int num_points = model->rowCount();

  std::string edge_str_to_be_pub = "";

  for( int i = 0; i < num_points; i++ )
  {
    QModelIndex child_index = model->index( i, 0 );
    rviz_common::properties::Property* child = model->getProp( child_index );
    rviz_common::properties::VectorProperty* subchild = (rviz_common::properties::VectorProperty*) child->childAt( 0 );
    Ogre::Vector3 vec = subchild->getVector();
    RCLCPP_DEBUG(rclcpp::get_logger("EdgeSelectionTool"), "%s, %.2f, %.2f, %.2f", child->getNameStd().c_str(), vec.x, vec.y, vec.z);
    //Marker pg_0_node/88, 30.49, -66.05, 1.24
    //Marker pg_0_edge_155/90, 0.00, 0.00, 0.00
    //Point 141488 [cloud 0x94354302778400], 41.52, -17.00, -1.88
    if (child->getNameStd().find("Marker") != std::string::npos) {
      if (child->getNameStd().find("edge") != std::string::npos) {

        std::stringstream edge_full_name(child->getNameStd());
        std::string segment;
        std::vector<std::string> seglist;

        while(std::getline(edge_full_name, segment, ' '))
        {
          seglist.push_back(segment);
        }
        RCLCPP_DEBUG(rclcpp::get_logger("EdgeSelectionTool"), "%s", seglist[1].c_str());
        edge_str_to_be_pub += seglist[1];
        edge_str_to_be_pub += ";";
      }
    }
  }
  std_msgs::msg::String tmp_str;
  tmp_str.data = edge_str_to_be_pub;
  selected_edge_pub_->publish(tmp_str);

  sel_manager->select(
    event.panel->getRenderWindow(),
    sel_start_x_,
    sel_start_y_,
    event.x,
    event.y,
    rviz_common::interaction::SelectionManager::Remove);

}

int EdgeSelectionTool::processKeyEvent(QKeyEvent * event, rviz_common::RenderPanel * panel)
{
  (void) panel;
  auto sel_manager = context_->getSelectionManager();

  if (event->key() == Qt::Key_F) {
    sel_manager->focusOnSelection();
  }

  if(event->type() == QKeyEvent::KeyPress)
  {
    std_msgs::msg::String tmp_str;
    tmp_str.data += event->key();
    
    if(event->key() == 'd' || event->key() == 'D')
    {
      selected_edge_pub_->publish(tmp_str);
    }
    return Render;
  }
  else
    return Render;

  return Render;
}

}  // namespace tools
}  // namespace dddmr_rviz_default_plugins

#include <pluginlib/class_list_macros.hpp>  // NOLINT
PLUGINLIB_EXPORT_CLASS(dddmr_rviz_default_plugins::tools::EdgeSelectionTool, rviz_common::Tool)
