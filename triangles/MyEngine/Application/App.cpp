/****************************************************************************************
 *
 *   App.cpp
 *
 *   Created by dmitry
 *   22.02.2021
 *
 ***/

//
/// App
///======================================================================================
/// the tinyxml2 library is used to parse the input scene data file.
/// Unlike the previous version, the displayed promises are inherited from the class
/// that the engine can display.
/// More details in the implementation files
///======================================================================================
///======================================================================================
//


#include "App.h"
#include "Entity.hpp"

#include <Octree/Octree.h>


#include <OtherLibs/random.h>


#define WINDOW_HEIGHT 1000
#define WINDOW_WIDTH  1000


namespace ezg
{


    AppLVL4::AppLVL4()
    {
        //-----------------------------------------------
        // create window
        std::vector< std::pair< int, int > > hints = {
                {GLFW_CLIENT_API, GLFW_NO_API},
                {GLFW_RESIZABLE,  GLFW_FALSE}
        };

        m_window = std::make_unique< Window >(WINDOW_WIDTH, WINDOW_HEIGHT
                                              , "LVL 4"
                                              , nullptr, nullptr
                                              , hints);
        m_window->setUserPointer(this);
        m_window->setKeyCallback(keyCallback);

        m_window->setInputMode(GLFW_STICKY_KEYS, 1);
        m_window->setInputMode(GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        const int wHeight = m_window->getHeight();
        const int wWidth = m_window->getWidth();
        m_window->setCursorPosition(wWidth / 2.0, wHeight / 2.0);
        // end create window
        //-----------------------------------------------

        m_driver = std::make_unique< engine::Engine >(*m_window);

        //-----------------------------------------------
        // init camera view
        m_cameraView.setAspect(static_cast< float >(wWidth) / wHeight);
        // end init camera view
        //-----------------------------------------------
    }

    AppLVL4::~AppLVL4()
    {
        std::transform(m_entities.begin(), m_entities.end(), m_entities.begin(), [&driver = m_driver](
                auto* pEnt) -> TriangleMesh*
        {
            driver->unload_object(*pEnt);
            delete pEnt;
            return nullptr;
        });
    }

    /****************************************************************************************
     *
     *   public methods
     *
     ***/


    tinyxml2::XMLError AppLVL4::loadSceneFromXML(const std::string& fileName)
    {
        tinyxml2::XMLDocument doc;
        auto res = doc.LoadFile(fileName.c_str());
        if (res != tinyxml2::XML_SUCCESS) {
            return res;
        }

        load_box_from_XML(doc);

        load_camera_from_XML(doc);

        load_objects_from_XML(doc);

        m_sceneIsLoaded = true;
        return tinyxml2::XML_SUCCESS;
    }


    void AppLVL4::run()
    {
        if (!m_sceneIsLoaded) {
            throw std::runtime_error("scene isn't loaded!");
        }

        for (auto* o : m_entities) {
            m_driver->upload_object(*o);
        }

        m_time.reset();

        Timer frameTimer;
        while (!m_window->shouldClose()) {
            float dTime = frameTimer.elapsed();
            frameTimer.reset();

            glfwPollEvents();

            update_entities_(dTime);

            update_camera_(dTime);
            m_driver->setCameraView(m_cameraView);

            m_driver->render_meshes(m_entities);
        }

        std::cout << "App closed!" << std::endl;
    }

    /***
     *
     *   public methods (end)
     *
     ****************************************************************************************/
    /****************************************************************************************
     *
     *   private methods
     *
     ***/


    void AppLVL4::load_box_from_XML(tinyxml2::XMLDocument& doc)
    {
        using tinyxml2p::getAttribute;

        auto* boxXML = doc.FirstChildElement("box");
        if (boxXML != nullptr) {
            float x1 = 0, x2 = 0, y1 = 0, y2 = 0, z1 = 0, z2 = 0;
            getAttribute(boxXML, "x1", x1, "y1", y1, "z1", z1);
            getAttribute(boxXML, "x2", x2, "y2", y2, "z2", z2);

            m_box.reup({x1, y1, z1}, {x2, y2, z2});
        }
    }

    void AppLVL4::load_camera_from_XML(tinyxml2::XMLDocument& doc)
    {
        using tinyxml2p::getAttribute;

        auto* cameraXML = doc.FirstChildElement("camera");
        if (cameraXML != nullptr) {
            m_speed = cameraXML->FloatAttribute("speed");

            auto* posXML = cameraXML->FirstChildElement("position");
            if (posXML != nullptr) {
                getAttribute(posXML, "x", m_cameraView.m_position.x
                             , "y", m_cameraView.m_position.y
                             , "z", m_cameraView.m_position.z);
            }

            auto* dirXML = cameraXML->FirstChildElement("direction");
            if (dirXML != nullptr) {
                getAttribute(dirXML, "x", m_cameraView.m_direction.x
                             , "y", m_cameraView.m_direction.y
                             , "z", m_cameraView.m_direction.z);
            }
        }
    }

    void AppLVL4::load_objects_from_XML(tinyxml2::XMLDocument& doc)
    {
        tinyxml2::XMLElement* objectXML = doc.FirstChildElement("objects");
        if (objectXML != nullptr) {
            objectXML = objectXML->FirstChildElement();
            while (objectXML != nullptr) {
                const std::string type = objectXML->Attribute("type");

                Entity* entity = nullptr;
                if (type == "mirror") {
                    entity = new Mirror;
                }
                else if (type == "justMesh") {
                    entity = new TriangleMesh;
                }
                else {
                    assert(0);
                }

                m_entities.push_back(dynamic_cast< engine::Renderable* >(entity));
                entity->loadFromXML(objectXML);

                objectXML = objectXML->NextSiblingElement();
            }
        }
    }


    void AppLVL4::update_entities_(float time)
    {
        for (auto* entity : m_entities) {
            auto* curEntity = dynamic_cast<Entity*>(entity);
            assert(entity != nullptr);

            //we detected the exit of the figure outside the box
            if (curEntity->m_position.x < m_box.getA().x || curEntity->m_position.x > m_box.getB().x) {
                curEntity->m_dirTravel.x = std::abs(curEntity->m_dirTravel.x)
                                           * ((m_box.getA().x - curEntity->m_position.x) /
                                              std::abs(m_box.getA().x - curEntity->m_position.x));
            }
            if (curEntity->m_position.y < m_box.getA().y || curEntity->m_position.y > m_box.getB().y) {
                curEntity->m_dirTravel.y = std::abs(curEntity->m_dirTravel.y)
                                           * ((m_box.getA().y - curEntity->m_position.y) /
                                              std::abs(m_box.getA().y - curEntity->m_position.y));
            }
            if (curEntity->m_position.z < m_box.getA().z || curEntity->m_position.z > m_box.getB().z) {
                curEntity->m_dirTravel.z = std::abs(curEntity->m_dirTravel.z)
                                           * ((m_box.getA().z - curEntity->m_position.z) /
                                              std::abs(m_box.getA().z - curEntity->m_position.z));
            }
            curEntity->update(time);
        }

    }


    void AppLVL4::update_camera_(float time)
    {
        int wHeight = m_window->getHeight(), wWidth = m_window->getWidth();

        if (m_window->getKey(GLFW_KEY_TAB) != GLFW_PRESS) {
            double x = m_window->getCursorXPos(), y = m_window->getCursorYPos();
            const float dx = m_prev_cursor_x - x;
            const float dy = m_prev_cursor_y - y;

            m_cameraView.turnInHorizontalPlane(glm::radians(90.f * dx / wWidth));
            m_cameraView.turnInVerticalPlane(glm::radians(90.f * dy / wHeight));

            m_prev_cursor_x = x;
            m_prev_cursor_y = y;
        }

        {
            bool forward = m_window->getKey(GLFW_KEY_W) == GLFW_PRESS;
            bool back = m_window->getKey(GLFW_KEY_S) == GLFW_PRESS;
            if (forward && back) { forward = back = false; }

            bool left = m_window->getKey(GLFW_KEY_A) == GLFW_PRESS;
            bool right = m_window->getKey(GLFW_KEY_D) == GLFW_PRESS;
            if (left && right) { left = right = false; }


            if (forward || back) {
                m_cameraView.moveAlongDirection(time * m_speed * ((forward) ? 1.f : -1.f));
            }
            if (left || right) {
                m_cameraView.movePerpendicularDirection(time * m_speed * ((right) ? 1.f : -1.f));
            }
        }

        {
            bool down = m_window->getKey(GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
            bool up = m_window->getKey(GLFW_KEY_SPACE) == GLFW_PRESS;
            if (down && up) { down = up = false; }

            if (down || up) {
                m_cameraView.moveVertical(time * m_speed * ((down) ? -1.f : 1.f));
            }
        }
    }


    /****************************************************************************************
     *
     *   window event response functions
     *
     ***/

    /*static*/ void AppLVL4::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
        auto app = static_cast<AppLVL4*>(glfwGetWindowUserPointer(window));

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        } else if (key == GLFW_KEY_TAB) {
            if (action == GLFW_PRESS) {
                app->m_window->setInputMode(GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            } else if (action == GLFW_RELEASE) {
                int wHeight = app->m_window->getHeight(), wWidth = app->m_window->getWidth();
                app->m_window->setCursorPosition(wWidth / 2.0, wHeight / 2.0);
                app->m_window->setInputMode(GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            }
        }
    }

    /***
     *
     *   window event response functions (end)
     *
     *********************************************************************************************/

}//namespace ezg