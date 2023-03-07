package fdse.microservice.repository;

import fdse.microservice.entity.Station;
import org.springframework.data.repository.CrudRepository;

import java.util.List;


public interface StationRepository extends CrudRepository<Station,String> {

    Station findByName(String name);

    Station findById(String id);

    @Override
    List<Station> findAll();
}
