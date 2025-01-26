/****************************************************************************************
 *
 *   App.h
 *
 *   Created by dmitry
 *   22.02.2021
 *
 ***/


//
// App
///======================================================================================
/// The application renders various triangular meshes, which can also be moved.
///
/// The Vulkan API is used for rendering.
/// Unlike the previous version (level 3), the written engine allows you to add new
/// objects after initialization. And it allows you to draw not all objects that have
/// been loaded, but only those that have been passed to the drawing function.
/// See engine implementation files for details
///======================================================================================
///======================================================================================
//


#pragma once

#include <vector>

#include <OtherLibs/timer.h>
#include <LAL/include/LAL.h>


#include <Engine/Engine.hpp>

#include <tinyxml2p.hpp>


namespace ezg
{

    class AppLVL4 final
    {
        std::unique_ptr< engine::Engine >    m_driver;

        std::unique_ptr< Window >    m_window;

        std::vector< engine::Renderable* > m_entities;
        la::Rectangle3               m_box;

        engine::CameraView                   m_cameraView;
        float                        m_speed = 100.f;

        Timer                        m_time;

        bool                         m_sceneIsLoaded = false;

        double m_prev_cursor_x = 0, m_prev_cursor_y = 0;

    public:

        AppLVL4           (const AppLVL4&) = delete; //not supported
        AppLVL4& operator=(const AppLVL4&) = delete; //not supported
        AppLVL4           (AppLVL4&&)      = delete; //not supported
        AppLVL4& operator=(AppLVL4&&)      = delete; //not supported

        AppLVL4();

        ~AppLVL4();


    public:

        /// loads scene information from a file.
        /// e.g. information about camera position & speed & direction, bounding box and objects
        /// \param fileName - file with data of scene
        /// \return result code(enumeration member). If successful, tinyxml2::XML_SUCCESS is returned
        tinyxml2::XMLError loadSceneFromXML(const std::string& fileName);

        bool sceneIsLoaded() const noexcept { return m_sceneIsLoaded; }

        /// Launch the app!
        /// A window opens with a visualization of the triangle meshes.
        /// Control returns after the window closes or after an emergency shutdown.
        /// \note inside the loops, the exit from which is performed by closing the window
        /// \return void
        void run();

    private:

        void load_box_from_XML(tinyxml2::XMLDocument& doc);
        void load_camera_from_XML(tinyxml2::XMLDocument& doc);
        void load_objects_from_XML(tinyxml2::XMLDocument& doc);
        void load_object_properties_from_XML(tinyxml2::XMLElement* elem);

        void update_entities_(float time);
        void update_camera_(float time);


        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
    };

}//namespace ezg