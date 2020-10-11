#include "../include/LALmath.h"

///////////////////////////////////////////////////////////////
//
// Vector & Vector
//
///////////////////////////////////////////////////////////////
namespace la
{
    Vector3f normalization(const Vector3f& _that)
    {
        if (!_that.isZero())
        {
            double m = _that.modul();
            return Vector3f(_that.x / m, _that.y / m, _that.z / m);
        }
        else
        {
            return Vector3f(0.f);
        }
    }

    double modul(const Vector3f& _v)
    {
        return std::sqrt(_v.x * _v.x + _v.y * _v.y + _v.z * _v.z);
    }

    bool codirected(const Vector3f& _lhs, const Vector3f& _rhs)
    {
        return collinear(_lhs, _rhs) && scalarProduct(_lhs, _rhs) >= 0.f;
    }

    Vector3f crossProduct(const Vector3f& _lhs, const Vector3f& _rhs)
    {
        auto res = Vector3f(_lhs.y * _rhs.z - _lhs.z * _rhs.y,
            _lhs.z * _rhs.x - _lhs.x * _rhs.z,
            _lhs.x * _rhs.y - _lhs.y * _rhs.x);

        return res;
    }


    double scalarProduct(const Vector3f& _lhs, const Vector3f& _rhs)
    {
        return _lhs.x * _rhs.x + _lhs.y * _rhs.y + _lhs.z * _rhs.z;
    }


    Vector3f operator -  (const Vector3f& _lhs, const Vector3f& _rhs) 
    { 
        return Vector3f(_lhs.x - _rhs.x, _lhs.y - _rhs.y, _lhs.z - _rhs.z); 
    }
    Vector3f operator +  (const Vector3f& _lhs, const Vector3f& _rhs) 
    { 
        return Vector3f(_lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z); 
    }
    Vector3f operator *  (const Vector3f& _lhs, const Vector3f& _rhs) 
    { 
        return crossProduct(_lhs, _rhs); 
    }
    Vector3f operator *  (const Vector3f& _lhs, double _n) 
    { 
        return { _lhs.x * _n, _lhs.y * _n, _lhs.z * _n }; 
    }
    double        operator ^  (const Vector3f& _lhs, const Vector3f& _rhs) 
    { 
        return scalarProduct(_lhs, _rhs); 
    }


//2D
    Vector2f normalization(const Vector2f& _that)
    {
        if (!_that.isZero())
        {
            double m = _that.modul();
            return Vector2f(_that.x / m, _that.y / m);
        }
        else
        {
            return Vector2f(0.f);
        }
    }

    double modul(const Vector2f& _v) 
    { 
        return std::sqrt(_v.x * _v.x + _v.y * _v.y); 
    }


    Vector3f crossProduct(const Vector2f& _lhs, const Vector2f& _rhs)
    {
        return crossProduct({ _lhs.x, _lhs.y, 0.f }, { _rhs.x, _rhs.y, 0.f });
    }

    double scalarProduct(const Vector2f& _lhs, const Vector2f& _rhs)
    {
        return (static_cast<double>(_lhs.x) * _rhs.x
            + static_cast<double>(_lhs.y) * _rhs.y);
    }


    Vector2f operator -  (const Vector2f& _lhs, const Vector2f& _rhs) 
    { 
        return Vector2f(_lhs.x - _rhs.x, _lhs.y - _rhs.y);
    }
    Vector2f operator +  (const Vector2f& _lhs, const Vector2f& _rhs) 
    { 
        return Vector2f(_lhs.x + _rhs.x, _lhs.y + _rhs.y);
    }
    double        operator ^  (const Vector2f& _lhs, const Vector2f& _rhs) 
    { 
        return scalarProduct(_lhs, _rhs);
    }
    Vector3f operator *  (const Vector2f& _lhs, const Vector2f& _rhs)
    { 
        return crossProduct(_lhs, _rhs);
    }
    Vector2f operator *  (const Vector2f& _lhs, double _n) 
    { 
        return { _lhs.x * _n, _lhs.y * _n };
    }

}//namespace la (Vector & Vector)



///////////////////////////////////////////////////////////////
//
// Vector & Line
//
///////////////////////////////////////////////////////////////
namespace la
{
//3D
    Vector3f projection(const Vector3f& _point, const Line3& _line)
    {
        Vector3f tmp_lv = normalization(_line.getV());

        double s = scalarProduct(tmp_lv, _point - _line.getP());

        return _line.getP() + tmp_lv * s;
    }


    double distance(const Vector3f& _point, const Line3& _line)
    {
        return (_point - projection(_point, _line)).modul();
    }

//2D
    Vector2f projection(const Vector2f& _point, const Line2& _line)
    {
        Vector2f tmp_lv = normalization(_line.getV());

        double s = scalarProduct(tmp_lv, _point - _line.getP());

        return _line.getP() + tmp_lv * s;
    }


    double distance(const Vector2f& _point, const Line2& _line)
    {
        return (_point - projection(_point, _line)).modul();
    }

}//namespace la (Vector & Line)



///////////////////////////////////////////////////////////////
//
// Line & Line
//
///////////////////////////////////////////////////////////////
namespace la
{
//3D
    std::pair<Vector3f, Intersec::quantity> findIntersection(const Line3& _lhs, const Line3& _rhs)
    {
        //Q1 and Q2 are points -> dot(Q2 - Q1, V1) = 0 & dot(Q2 - Q1, V2) = 0
        //Q1 = P1 + S1 * V1
        //Q2 = P2 + S2 * V2
        //
        //...
        //
        //S1 * dot(V1, V1) - S2 * dot(V2, V1) = dot(P2 - P1, V1)
        //S1 * dot(V1, V2) - S2 * dot(V2, V2) = dot(P2 - P1, V2)
        //<=>
        //S1 * a - S2 * b = c
        //S2 * b - S2 * d = f

        Vector3f pp = _rhs.getP() - _lhs.getP();
        double a = scalarProduct(_lhs.getV(), _lhs.getV());
        double b = scalarProduct(_lhs.getV(), _rhs.getV());
        double c = scalarProduct(pp, _lhs.getV());
        double d = scalarProduct(_rhs.getV(), _rhs.getV());
        double f = scalarProduct(pp, _rhs.getV());
        double det = -a * d + b * b;

        std::pair<Vector3f, Intersec::quantity> res(0.f, Intersec::quantity::Nop);
        if (std::abs(det) < EPSILON)
        {
            //V1 || V2

            //V1 == V2
            if ((_lhs.getP() - _rhs.getP()).collinear(_lhs.getV()))
            {
                res.second = Intersec::quantity::Same;
            }
        }
        else
        {
            double det1 = -d * c + b * f;
            double det2 = a * f - b * c;

            double S1 = det1 / det;
            double S2 = det2 / det;

            Vector3f Q1 = _lhs.getP() + _lhs.getV() * S1;
            Vector3f Q2 = _rhs.getP() + _rhs.getV() * S2;

            if (Q1 == Q2)
            {
                res.first = Q1;
                res.second = Intersec::quantity::One;
            }
        }

        return res;
    }

    double distance(const Line3& _lhs, const Line3& _rhs)
    {
        //Q1 and Q2 are points -> dot(Q2 - Q1, V1) = 0 & dot(Q2 - Q1, V2) = 0
        //Q1 = P1 + S1 * V1
        //Q2 = P2 + S2 * V2
        //
        //...
        //
        //S1 * dot(V1, V1) - S2 * dot(V2, V1) = dot(P2 - P1, V1)
        //S1 * dot(V1, V2) - S2 * dot(V2, V2) = dot(P2 - P1, V2)
        //<=>
        //S1 * a - S2 * b = c
        //S2 * b - S2 * d = f

        Vector3f pp = _rhs.getP() - _lhs.getP();
        double a = scalarProduct(_lhs.getV(), _lhs.getV());
        double b = scalarProduct(_lhs.getV(), _rhs.getV());
        double c = scalarProduct(pp, _lhs.getV());
        double d = scalarProduct(_rhs.getV(), _rhs.getV());
        double f = scalarProduct(pp, _rhs.getV());
        double det = -a * d + b * b;

        double res = 0.f;
        if (std::abs(det) < EPSILON)
        {
            //V1 || V2

            //V1 == V2
            if ((_lhs.getP() - _rhs.getP()).collinear(_lhs.getV()))
            {
                res = 0.f;
            }
            else
            {
                const double S2 = 0.f;
                const double S1 = c / a;

                res = (_lhs.getP() + _lhs.getV() * S1 - _rhs.getP()).modul();
            }
        }
        else
        {
            const double det1 = -d * c + b * f;
            const double det2 = a * f - b * c;

            const double S1 = det1 / det;
            const double S2 = det2 / det;

            const Vector3f Q1 = _lhs.getP() + _lhs.getV() * S1;
            const Vector3f Q2 = _rhs.getP() + _rhs.getV() * S2;

            res = (Q1 - Q2).modul();
        }

        return res;
    }

//2D
    std::pair<Vector2f, Intersec::quantity> findIntersection(const Line2& _lhs, const Line2& _rhs)
    {
        if (!_lhs.intersection(_rhs)) {
            return { {}, Intersec::quantity::Nop };
        }

        std::pair<Vector2f, Intersec::quantity> res({}, Intersec::quantity::Nop);

        if (!_lhs.getV().collinear(_rhs.getV())) {
            const double t = crossProduct(_rhs.getP() - _lhs.getP(), _rhs.getV()).z
                / crossProduct(_lhs.getV(), _rhs.getV()).z;
            res.first = _lhs.getP() + t * _lhs.getV();
            res.second = Intersec::quantity::One;
        }
        else {
            res.first = _lhs.getP();
            res.second = Intersec::quantity::Same;
        }

        return res;
    }

}//namespace la (Line & Line)



///////////////////////////////////////////////////////////////
//
// LineSegment & LineSegment
//
///////////////////////////////////////////////////////////////
namespace la
{
//3D
    std::pair<LineSegment3, Intersec::quantity>
        findIntersection(const LineSegment3& _lhs, const LineSegment3& _rhs)
    {
        const std::pair<Vector3f, Intersec::quantity> intersec_line = findIntersection(_lhs.toLine(), _rhs.toLine());

        std::pair<LineSegment3, Intersec::quantity> res({}, Intersec::quantity::Nop);
        if (intersec_line.second == Intersec::quantity::Nop) 
        {
            /*nop*/
        }
        else if (intersec_line.second == Intersec::quantity::One)
        {
            if (_lhs.contein(intersec_line.first) && _rhs.contein(intersec_line.first))
            {
                res.first = LineSegment3(intersec_line.first, intersec_line.first, LineSegment3::Type::TwoPoints);
                res.second = Intersec::quantity::One;
            }
        }
        else if (intersec_line.second == Intersec::quantity::Same)
        {
            double s0 = 0;
            double s1 = 0;
            if (_lhs.getV().modul() > EPSILON) {
                s0 = scalarProduct(normalization(_lhs.getV()), _rhs.getP() - _lhs.getP())
                    / _lhs.getV().modul();
                s1 = scalarProduct(normalization(_lhs.getV()), _rhs.getP() + _rhs.getV() - _lhs.getP())
                    / _lhs.getV().modul();
            }

            const auto inter_res = findIntersection({ 0.f, 1.f }, { s0, s1 });
            if (inter_res.second != Intersec::quantity::Nop)
            {
                res.second = inter_res.second;

                res.first = LineSegment3(_lhs.getP() + _lhs.getV() * inter_res.first.getA()
                    , _lhs.getP() + _lhs.getV() * inter_res.first.getB());
            }
        }
        return res;
    }


//2D
    std::pair<LineSegment2, Intersec::quantity>
        findIntersection(const LineSegment2& _lhs, const LineSegment2& _rhs)
    {
        const std::pair<Vector2f, Intersec::quantity> intersec_line = findIntersection(_lhs.toLine(), _rhs.toLine());

        std::pair<LineSegment2, Intersec::quantity> res({}, Intersec::quantity::Nop);
        if (intersec_line.second == Intersec::quantity::Nop) {
            /*nop*/
        }
        else if (intersec_line.second == Intersec::quantity::One)
        {
            if (_lhs.contein(intersec_line.first) && _rhs.contein(intersec_line.first))
            {
                res.first = LineSegment2(intersec_line.first, intersec_line.first, LineSegment2::Type::TwoPoints);
                res.second = Intersec::quantity::One;
            }
        }
        else if (intersec_line.second == Intersec::quantity::Same)
        {
            const double s0 = scalarProduct(normalization(_lhs.getV()), _rhs.getP() - _lhs.getP()) / _lhs.getV().modul();
            const double s1 = scalarProduct(normalization(_lhs.getV()), _rhs.getP() + _rhs.getV() - _lhs.getP()) / _lhs.getV().modul();

            const auto inter_res = findIntersection({ 0.f, 1.f }, { s0, s1 });
            if (inter_res.second != Intersec::quantity::Nop)
            {
                res.second = inter_res.second;

                res.first = LineSegment2(_lhs.getP() + _lhs.getV() * inter_res.first.getA(), _lhs.getP() + _lhs.getV() * inter_res.first.getB());
            }
        }
        return res;
    }

//1D
    std::pair<LineSegment1, Intersec::quantity>
        findIntersection(const LineSegment1& _lhs, const LineSegment1& _rhs)
    {
        auto res = std::make_pair(LineSegment1(), Intersec::quantity::Nop);

        if (!_lhs.intersection(_rhs)) {
            return res;
        }

        double res_a = 0.f;
        double res_b = 0.f;
        // m_a === a1
        // m_b === b1
        // ...
        //      a2*-----------*b2
        //  a1*--------*b1
        if (_lhs.getB() > _rhs.getA())
        {
            if (_lhs.getA() < _rhs.getB()) {
                if (_lhs.getA() < _rhs.getA())
                    res_a = _rhs.getA();
                else
                    res_a = _lhs.getA();

                if (_lhs.getB() > _rhs.getB())
                    res_b = _rhs.getB();
                else
                    res_b = _lhs.getB();
                if (res_a != res_b) {
                    res.second = Intersec::quantity::Interval;
                }
                else {
                    res.second = Intersec::quantity::One;
                }
            }
            else {
                //a2*---------*b2
                //          a1*-------------*b1
                res_a = res_b = _lhs.getA();
                res.second = Intersec::quantity::One;
            }
        }
        else if (_lhs.getB() == _rhs.getA()) {
            //              a2*---------*b2
            //a1*-------------*b1
            res_a = res_b = _lhs.getB();
            res.second = Intersec::quantity::One;
        }


        res.first = LineSegment1(res_a, res_b);

        return res;
    }


}//namespace la (LineSegment & LineSegment)


///////////////////////////////////////////////////////////////
//
// LineSegment & LineSegment
//
///////////////////////////////////////////////////////////////
namespace la
{
    //3D
    std::pair<Vector3f, Intersec::quantity>
        findIntersection(const LineSegment3& _ls, const Line3& _line)
    {
        auto result = std::make_pair(Vector3f(0.f), Intersec::quantity::Nop);

        const auto tmp_res = findIntersection(_ls.toLine(), _line);

        if (tmp_res.second == Intersec::quantity::One)
        {
            const double s = (tmp_res.first - _ls.getP()).modul() / _ls.getV().modul();

            if (s >= 0.f && s <= 1.f)
            {
                result.first = tmp_res.first;
                result.second = Intersec::quantity::One;
            }
        }
        else if (tmp_res.second == Intersec::quantity::Same)
        {
            result.second = Intersec::quantity::Same;
        }

        return result;
    }
}//namespace la (LineSegment & Line)


///////////////////////////////////////////////////////////////
//
// Plane & Plane
//
///////////////////////////////////////////////////////////////
namespace la
{
    bool intersec(const Plane& _lhs, const Plane& _rhs)
    {
        return intersec(_lhs, Line3(_rhs.getP(), _rhs.getA(), Line3::Type::PointAndVector))
            || intersec(_lhs, Line3(_rhs.getP(), _rhs.getB(), Line3::Type::PointAndVector));
    }

    std::pair<Line3, Intersec::quantity> findIntersec(const Plane& _lhs, const Plane& _rhs)
    {
        auto res = std::make_pair(Line3(), Intersec::quantity::Nop);

        if (intersec(_lhs, _rhs))
        {
            //r = E + k * V - intersection line
            //
            //V = [n1 * n2];
            //
            //dot(P2 - E, n2) = 0
            //dot(P1 - E, n1) = 0
            //E = a * n1 + b * n2;
            //<=>
            //a * dot(n1, n1) + b * dot(n2, n1) = dot(p1, n1)
            //a * dot(n2, n1) + b * dot(n2, n2) = dot(p2, n2)
            //<=>
            //a * q + b * w = s1
            //a * w + b * l = s2
            //<=>
            //det = q * l - w^2
            //det1 = s1 * l - s2 * w
            //det2 = q * s2 - s1 * w
            //...

            const double q = scalarProduct(_lhs.getN(), _lhs.getN());
            const double w = scalarProduct(_lhs.getN(), _rhs.getN());
            const double l = scalarProduct(_rhs.getN(), _rhs.getN());
            const double s1 = scalarProduct(_lhs.getP(), _lhs.getN());
            const double s2 = scalarProduct(_rhs.getP(), _rhs.getN());

            const double det = q * l - w * w;
            if (std::abs(det) < EPSILON)
            {
                res.second = Intersec::quantity::Same;
                assert(_lhs == _rhs);
            }
            else
            {
                const double det1 = s1 * l - w * s2;
                const double det2 = q * s2 - w * s1;

                const double a = det1 / det;
                const double b = det2 / det;

                const Vector3f e = _lhs.getN() * a + _rhs.getN() * b;
                const Vector3f v = crossProduct(normalization(_lhs.getN()), normalization(_rhs.getN()));
                res.first.reup(e, v, Line3::Type::PointAndVector);
                res.second = Intersec::quantity::Interval;
            }
        }

        return res;
    }

    bool equal(const Plane& _lhs, const Plane& _rhs)
    {
        bool res = true;

        res = res && _lhs.getN() * _rhs.getN() == Vector3f(0.f);
        res = res && std::abs((_lhs.getP() - _rhs.getP()) ^ _lhs.getN()) < EPSILON;

        return res;
    }

    double distance(const Plane& _lhs, const Plane& _rhs)
    {
        return std::abs(distanceWithSign(_lhs, _rhs));
    }

    double distanceWithSign(const Plane& _lhs, const Plane& _rhs)
    {
        double res = 0.f;

        if (!intersec(_lhs, _rhs))
        {
            //dot(_rhs.getP() - (_lhs.getN() * s + _lhs.getP()), _rhs.getN()) = 0
            //dot(_rhs.getP(), _rhs.getN()) - dot(_lhs.getN() * s, _rhs.getN()) 
            //    - dot(_lhs.getP(), _rhs.getN()) = 0
            //
            // s * dot(_lhs.getN(), _rhs.getN()) =  dot(_rhs.getP(), _rhs.getN()) 
            //                                      - dot(_lhs.getP(), _rhs.getN())
            const double a = scalarProduct(_lhs.getN(), _rhs.getN());
            const double b = scalarProduct(_rhs.getP(), _rhs.getN());
            const double c = scalarProduct(_lhs.getP(), _rhs.getN());

            res = (b - c) / a;
        }

        return res;
    }

}//namespace la (Plane & Plane)


///////////////////////////////////////////////////////////////
//
// Plane & Vector
//
///////////////////////////////////////////////////////////////
namespace la
{
//3D
    bool contein(const Plane& _pl, const Vector3f& _point)
    {
        return std::abs(scalarProduct(_pl.getP() - _point, _pl.getN())) < EPSILON;
    }

    Vector3f projection(const Vector3f& _point, const Plane& _pl)
    {
        const Vector3f res = _point;

        const double s = distanceWithSign(_pl, _point);


        return _point - _pl.getN() * s;
    }

    double distance(const Plane& _pl, const Vector3f _point)
    {
        return distance(_pl, Plane(_point, _pl.getA(), _pl.getB()));
    }

    double distanceWithSign(const Plane& _pl, const Vector3f _point)
    {
        return distanceWithSign(_pl, Plane(_point, _pl.getA(), _pl.getB()));
    }

}//namespace la (Plane & Vector)


///////////////////////////////////////////////////////////////
//
// Plane & Line
//
///////////////////////////////////////////////////////////////
namespace la
{
    bool intersec(const Plane& _pl, const Line3& _ln)
    {
        bool res = false;

        if (contein(_pl, _ln))
        {
            res = true;
        }
        else
        {
            res = std::abs(scalarProduct(_pl.getN(), _ln.getV())) > EPSILON;
        }

        return res;
    }

    Line3 projection(const Line3& _line, const Plane& _pl)
    {
        const Vector3f p1_ = projection(_line.getP(), _pl);
        const Vector3f p2_ = projection(_line.getP() + _line.getV(), _pl);

        return {p1_, p2_, Line3::Type::TwoPoints};
    }

    std::pair<Vector3f, Intersec::quantity>
        findIntersec(const Plane& _pl, const Line3& _ln)
    {
        auto res = std::make_pair(Vector3f(0.f), Intersec::quantity::Nop);

        if (intersec(_pl, _ln)) {
            //Q - required point
            //
            //Q = _ln.getP() + s * _ln.getV();
            //dot(_pl.getN(), Q - _pl.getP()) = 0;
            //<=>
            //dot(_pl.getN(), _ln.getP() + s * _ln.getV()) - dot(_pl.getN(), _pl.getP()) = 0;
            //<=>
            //s = [ dot(_pl.getN(), _pl.getP()) - dot(_pl.getN(), _ln.getP()) ] / dot(_pl.getN(), _ln.getV());
            //<=> 
            //s = (a - b) / c;

            const double c = _pl.getN() ^ _ln.getV();

            if (std::abs(c) < EPSILON)
            {
                res.second = Intersec::quantity::Same;
            }
            else
            {
                res.second = Intersec::quantity::One;

                const double a = _pl.getN() ^ _pl.getP();
                const double b = _pl.getN() ^ _ln.getP();

                const double s = (a - b) / c;
                res.first = _ln.getP() + s * _ln.getV();
            }
        }
        
        return res;
    }

    double distanceWithSign(const Plane& _pl, const Line3 _ln)
    {
        double res = 0.f;

        if (!_ln.getV().collinear(_pl.getA()))
        {
            res = distanceWithSign(_pl, Plane(_ln.getP(), _ln.getV(), _pl.getA()));
        }
        else
        {
            res = distanceWithSign(_pl, Plane(_ln.getP(), _ln.getV(), _pl.getB()));
        }

        return res;
    }

    double distance(const Plane& _pl, const Line3 _ln)
    {
        return std::abs(distanceWithSign(_pl, _ln));
    }

    bool contein(const Plane& _pl, const Line3& _ln)
    {
        return contein(_pl, _ln.getP()) && contein(_pl, _ln.getP() + _ln.getV());
    }

}//namespace la (Plane & Line)


///////////////////////////////////////////////////////////////
//
// Triangle & Triangle
//
///////////////////////////////////////////////////////////////
namespace la
{
    //3D
    bool intersec(const Triangle& _lhs, const Triangle& _rhs)
    {
        bool result = false;

        //2D
        if (_lhs.getPlane() == _rhs.getPlane())
        {
            result = result || _lhs.contein(_rhs.getA());
            result = result || _lhs.contein(_rhs.getB());
            result = result || _lhs.contein(_rhs.getC());
        }
        //3D
        else if (intersec(_lhs.getPlane(), _rhs.getPlane()))
        {
            const auto line_inters = findIntersec(_lhs.getPlane(), _rhs.getPlane());
            assert(line_inters.second != Intersec::quantity::Nop);
            assert(line_inters.second != Intersec::quantity::Same);

            const auto res1 = findIntersec(_lhs, line_inters.first);
            if (res1.second != Intersec::quantity::Nop)
            {
                const auto res2 = findIntersec(_rhs, line_inters.first);
                if (res2.second != Intersec::quantity::Nop)
                {
                    const auto findRes = findIntersection(res1.first, res2.first);
                    if (findRes.second != Intersec::quantity::Nop) {
                        result = true;
                    }
                }
            }
        }
        else
        {
            result = false;
        }
        
        return result;
    }

}//namespace la (Triangle & Triangle)


///////////////////////////////////////////////////////////////
//
// Triangle & Line
//
///////////////////////////////////////////////////////////////
namespace la
{
    std::pair<LineSegment3, Intersec::quantity> findIntersec(const Triangle& _tr, const Line3& _line)
    {
        const auto line_plane = findIntersec(_line, _tr.getPlane());
        auto result = std::make_pair(LineSegment3(), Intersec::quantity::Nop);

        if (line_plane.second == Intersec::quantity::One)
        {
            if (_tr.contein(line_plane.first))
            {
                result.first.reup(line_plane.first, line_plane.first, LineSegment3::Type::TwoPoints);
                result.second = Intersec::quantity::One;
            }
        }
        else if (line_plane.second == Intersec::quantity::Same)
        {
            const Vector3f a_shtrix = projection(_tr.getA(), _line);
            const Vector3f b_shtrix = projection(_tr.getB(), _line);
            const Vector3f c_shtrix = projection(_tr.getC(), _line);

            const Vector3f ash_to_a = _tr.getA() - a_shtrix;
            const Vector3f bsh_to_b = _tr.getB() - b_shtrix;
            const Vector3f csh_to_c = _tr.getC() - c_shtrix;

            const double a_dist = (ash_to_a).modul();
            const double b_dist = (bsh_to_b).modul() * ((codirected(ash_to_a, bsh_to_b) ? 1 : -1));
            const double c_dist = (csh_to_c).modul() * ((codirected(ash_to_a, csh_to_c) ? 1 : -1));

            const double a_mdist = std::abs(a_dist);
            const double b_mdist = std::abs(b_dist);
            const double c_mdist = std::abs(c_dist);

            if (a_dist * b_dist * c_dist > EPSILON)
            {
                /*nop*/
            }
            else if (std::abs(a_dist) < EPSILON)
            {
                if (std::abs(b_dist) < EPSILON)
                {
                    assert(std::abs(c_dist) > EPSILON);

                    result.first.reup(_tr.getA(), _tr.getB(), LineSegment3::Type::TwoPoints);
                    result.second = Intersec::quantity::Interval;
                }
                else if (std::abs(c_dist) < EPSILON)
                {
                    result.first.reup(_tr.getA(), _tr.getC(), LineSegment3::Type::TwoPoints);
                    result.second = Intersec::quantity::Interval;
                }
                else if (c_dist * b_dist > 0.f)
                {
                    result.first.reup(_tr.getA(), _tr.getA());
                    result.second = Intersec::quantity::One;
                }
                else
                {
                    result.first.reup(a_shtrix, b_shtrix + (c_shtrix - b_shtrix) * (b_mdist / (c_mdist + b_mdist)));
                    result.second = Intersec::quantity::Interval;
                }
            }
            else if (std::abs(b_dist) < EPSILON)
            {
                if (std::abs(c_dist) < EPSILON)
                {
                    result.first.reup(_tr.getB(), _tr.getC(), LineSegment3::Type::TwoPoints);
                    result.second = Intersec::quantity::Interval;
                }
                else if (c_dist * a_dist > 0.f)
                {
                    result.first.reup(_tr.getB(), _tr.getB());
                    result.second = Intersec::quantity::One;
                }
                else
                {
                    result.first.reup(b_shtrix, a_shtrix + (c_shtrix - a_shtrix) * (a_mdist / (c_mdist + a_mdist)));
                    result.second = Intersec::quantity::Interval;
                }
            }
            else if (std::abs(c_dist) < EPSILON)
            {
                if (a_dist * b_dist > 0.f)
                {
                    result.first.reup(_tr.getC(), _tr.getC());
                    result.second = Intersec::quantity::One;
                }
                else
                {
                    result.first.reup(c_shtrix, b_shtrix + (a_shtrix - b_shtrix) * (b_mdist / (a_mdist + b_mdist)));
                    result.second = Intersec::quantity::Interval;
                }
            }
            else
            {
                result.second = Intersec::quantity::Interval;

                if (a_dist * b_dist > 0.f)
                {
                    const auto res_ca = findIntersection(_line, LineSegment3(_tr.getC(), _tr.getA()));
                    const auto res_cb = findIntersection(_line, LineSegment3(_tr.getC(), _tr.getB()));

                    result.first.reup(res_ca.first, res_cb.first);
                }
                else if (b_dist * c_dist > 0.f)
                {
                    const auto res_ab = findIntersection(_line, LineSegment3(_tr.getA(), _tr.getB()));
                    const auto res_ac = findIntersection(_line, LineSegment3(_tr.getA(), _tr.getC()));

                    result.first.reup(res_ab.first, res_ac.first);
                }
                else if (c_dist * a_dist > 0.f)
                {
                    const auto res_bc = findIntersection(_line, LineSegment3(_tr.getB(), _tr.getC()));
                    const auto res_ba = findIntersection(_line, LineSegment3(_tr.getB(), _tr.getA()));

                    result.first.reup(res_ba.first, res_bc.first);
                }
                else { assert(0); }
            }
        }
        else if (line_plane.second != Intersec::quantity::Nop) { assert(0); }

        return result;
    }


}//namespace la (Triangle & Line)